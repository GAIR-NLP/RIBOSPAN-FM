# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cosine endpoints and masked accuracy with attention held to the pretraining span.

A model pretrained on at most ``context_window`` tokens never saw a query and a
key farther apart than that. This stage masks every such pair at inference, in
every attention layer, and repeats the three cosine endpoints and the
masked-nucleotide accuracy. Weights and position encodings are left as they are.

Every registry model is run, each held to its own span. For a model pretrained
on the longest length group the mask removes almost nothing. HydraRNA has
attention in two of its twelve layers; its state-space layers are not held.

The models listed under ``restricted_attention.span_sweep`` are also scored at
each listed span shorter than their own, which shows how much of their
accuracy rests on attention beyond that distance.
"""

from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cosine import CsvSink, _as_bool, _job_metric_seed, _pair_metric_seed
from .metrics import context_cosine_rows, diffusion_cosine_rows
from .mlm import (
    MLM_DEFINITIONS,
    build_parser,
    masked_accuracy,
    masked_positions,
    prepare_stage,
    start_metadata,
    write_metadata,
)
from .model_io import ModelSpec, load_encoder


SPAN_BACKENDS = ("rnabert", "ribospan", "rnaverse", "ribogram", "hydrarna")
HF_ATTENTION_CLASSES = {
    "RNABertSelfAttention",
    "RiboSpanSelfAttention",
    "RNAverseSelfAttention",
}
RESTRICTED_FIELDS = [
    "model",
    "model_path",
    "pair_group_id",
    "structured_job_id",
    "native_job_id",
    "seed",
    "length",
    "length_group",
    "attention_span",
    "delta_context_separation_structured_vs_native",
    "cross_context_same_base_cos_structured",
    "relative_distal_075",
    "mask_rate",
    "masked_positions",
    "mlm_accuracy",
]
SPAN_SWEEP_FILE = "attention_span_sweep_units.csv"

# ``span`` is the largest allowed |query - key| distance in tokens; None leaves
# attention unrestricted. The band is rebuilt only when the sequence changes.
_STATE: dict[str, Any] = {"span": None, "band": None, "band_key": None}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse experiment arguments; paths never default relative to the CWD."""

    return build_parser(
        "Repeat the cosine endpoints and masked accuracy with attention held to "
        "each model's pretraining span."
    ).parse_args(argv)


@contextmanager
def attention_span(span: int | None) -> Iterator[None]:
    """Hold patched attention to ``|query - key| <= span`` inside the block."""

    previous = _STATE["span"]
    _STATE["span"] = span
    try:
        yield
    finally:
        _STATE["span"] = previous
        _STATE["band"] = None
        _STATE["band_key"] = None


def _band(tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive ``(tokens, tokens)`` mask: 0 inside the span, -inf outside."""

    key = (tokens, _STATE["span"], str(device), str(dtype))
    if _STATE["band_key"] != key:
        index = torch.arange(tokens, device=device)
        outside = (index[:, None] - index[None, :]).abs() > _STATE["span"]
        band = torch.zeros((tokens, tokens), device=device, dtype=dtype)
        band.masked_fill_(outside, float("-inf"))
        _STATE["band"] = band
        _STATE["band_key"] = key
    return _STATE["band"]


def _patch_additive_mask_attention(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """RNABert, RiboSpan, and RNAverse add ``attention_mask`` to the logits."""

    patched = 0
    for module in model.modules():
        if type(module).__name__ not in HF_ATTENTION_CLASSES:
            continue
        original = module.forward

        def forward(
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            *args: Any,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if _STATE["span"] is not None:
                band = _band(hidden_states.shape[1], hidden_states.device, dtype)
                band = band[None, None]
                attention_mask = (
                    band
                    if attention_mask is None
                    else attention_mask.to(dtype) + band
                )
            return _original(hidden_states, attention_mask, *args, **kwargs)

        module.forward = forward
        patched += 1
    return patched


def _patch_ribogram_attention(model: torch.nn.Module) -> int:
    """RiboGRAM attention takes no additive mask, so the banded pass is spelled out."""

    patched = 0
    for layer in model.layers:
        attention = layer.self_attn
        original = attention.forward

        def forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            bias: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
            need_head_weights: bool = False,
            _module: Any = attention,
            _original: Any = original,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if _STATE["span"] is None:
                return _original(
                    query=query,
                    key=key,
                    value=value,
                    bias=bias,
                    key_padding_mask=key_padding_mask,
                    need_head_weights=need_head_weights,
                )
            if key_padding_mask is not None or need_head_weights:
                raise RuntimeError(
                    "Restricted RiboGRAM attention expects one unpadded sequence"
                )
            length, batch, width = query.shape
            shape = (length, batch * _module.num_heads, _module.head_dim)
            q = (_module.q_proj(query) * _module.scaling).contiguous().view(*shape)
            k = _module.k_proj(key).contiguous().view(*shape)
            v = _module.v_proj(value).contiguous().view(*shape)
            q, k = _module.rot_emb(q.transpose(0, 1), k.transpose(0, 1))
            logits = torch.bmm(q, k.transpose(1, 2))
            logits = logits + _band(length, logits.device, logits.dtype)[None]
            weights = F.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            del logits
            output = torch.bmm(weights, v.transpose(0, 1))
            del weights
            output = output.transpose(0, 1).contiguous().view(length, batch, width)
            return _module.out_proj(output), None

        attention.forward = forward
        patched += 1
    return patched


def _patch_flash_attention(model: torch.nn.Module) -> int:
    """HydraRNA's attention layers call FlashAttention, which takes no additive mask."""

    patched = 0
    for module in model.modules():
        if type(module).__name__ != "FlashSelfAttention":
            continue
        original = module.forward

        def forward(
            qkv: torch.Tensor,
            causal: bool | None = None,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: int | None = None,
            _module: Any = module,
            _original: Any = original,
        ) -> torch.Tensor:
            if _STATE["span"] is None:
                return _original(
                    qkv, causal=causal, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                )
            if cu_seqlens is not None or causal or _module.causal:
                raise RuntimeError(
                    "Restricted FlashAttention expects one unpadded bidirectional "
                    "sequence"
                )
            # qkv is (batch, tokens, 3, heads, head_dim).
            query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
            band = _band(qkv.shape[1], qkv.device, qkv.dtype)[None, None]
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=band, scale=_module.softmax_scale
            )
            return output.transpose(1, 2)

        module.forward = forward
        patched += 1
    return patched


def restrict_attention(loaded: Any) -> int:
    """Make every attention layer of ``loaded`` obey ``attention_span``.

    Returns the number of patched layers. Zero means the backend has no
    attention layer this stage knows how to mask.
    """

    backend = (loaded.spec.backend or "rnabert").lower()
    if backend == "ribogram":
        return _patch_ribogram_attention(loaded.model)
    if backend == "hydrarna":
        return _patch_flash_attention(loaded.model)
    if backend in SPAN_BACKENDS:
        return _patch_additive_mask_attention(loaded.model, loaded.dtype)
    return 0


def _metric_settings(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The context and diffusion settings the cosine stage resolves in ``run``."""

    cosine_config = config["cosine"]
    sampling = config["sampling"]
    context = dict(cosine_config.get("context", {}))
    diffusion = dict(cosine_config.get("diffusion", {}))
    context_settings = {
        "max_positions_per_context": int(context.get("max_positions_per_context", 512)),
        "max_pairs_per_metric": int(sampling["context_pair_sample_size"]),
        "anisotropy_pairs": int(sampling["anisotropy_pair_sample_size"]),
        "anisotropy_match_base": _as_bool(
            context.get("anisotropy_match_base", True),
            "context.anisotropy_match_base",
        ),
        "motif_interior_trim": int(context.get("motif_interior_trim", 0)),
        "background_exclusion_radius": int(
            context.get("background_exclusion_radius", 0)
        ),
    }
    diffusion_settings = {
        "distance_bins": sampling["representation_distance_bins"],
        "far_field_threshold": float(sampling["far_field_threshold"]),
        "relative_distal_threshold": float(
            diffusion.get("relative_distal_threshold", 0.75)
        ),
        "relative_distal_thresholds": diffusion.get(
            "relative_distal_thresholds", [0.25, 0.50, 0.75]
        ),
        "max_positions_per_bin": int(sampling["max_positions_per_bin"]),
    }
    return context_settings, diffusion_settings


def _final_layer_endpoints(
    loaded: Any,
    structured: Mapping[str, Any],
    native: Mapping[str, Any],
    context_settings: Mapping[str, Any],
    diffusion_settings: Mapping[str, Any],
) -> dict[str, float]:
    """ΔCS, C_cross, and D_distal of one pair, with the cosine stage's seeds."""

    hidden: dict[str, tuple[torch.Tensor, ...]] = {}
    separation: dict[str, np.ndarray] = {}
    cross: dict[str, np.ndarray] = {}
    baselines: dict[str, list[float]] = {}
    for job in (structured, native):
        control = str(job["control_type"])
        hidden[control] = loaded.hidden_states(str(job["sequence"]))[-1:]
        rows, baselines[control] = context_cosine_rows(
            hidden[control],
            str(job["sequence"]),
            int(job["motif_start"]),
            int(job["motif_end"]),
            seed=_job_metric_seed(job, "context"),
            **context_settings,
        )
        raw = sorted(
            (row for row in rows if row["metric_variant"] == "raw"),
            key=lambda row: row["base"],
        )
        separation[control] = np.asarray(
            [row["context_separation"] for row in raw], dtype=float
        )
        cross[control] = np.asarray(
            [row["cross_context_same_base_cos"] for row in raw], dtype=float
        )
    rows = diffusion_cosine_rows(
        hidden["structured"],
        hidden["native"],
        str(structured["sequence"]),
        str(native["sequence"]),
        int(structured["motif_start"]),
        int(structured["motif_end"]),
        motif_anisotropy_baselines=baselines["structured"],
        control_anisotropy_baselines=baselines["native"],
        seed=_pair_metric_seed(structured, native, "diffusion"),
        **diffusion_settings,
    )
    distal = next(row for row in rows if row["metric_variant"] == "raw")
    return {
        "delta_context_separation_structured_vs_native": float(
            np.nanmean(separation["structured"] - separation["native"])
        ),
        "cross_context_same_base_cos_structured": float(
            np.nanmean(cross["structured"])
        ),
        "relative_distal_075": float(distal["relative_distal_075"]),
    }


def _restriction(spec: ModelSpec) -> tuple[bool, str]:
    backend = (spec.backend or "rnabert").lower()
    if backend not in SPAN_BACKENDS:
        return False, f"{backend} has no attention span this stage can hold"
    return True, ""


def _span_sweep(
    config: Mapping[str, Any], restrictable: Mapping[str, ModelSpec]
) -> dict[str, list[int]]:
    """The spans below its own at which each configured model is scored again."""

    settings = (config.get("restricted_attention") or {}).get("span_sweep") or {}
    spans = sorted(int(span) for span in settings.get("spans", ()))
    sweep: dict[str, list[int]] = {}
    for name in settings.get("models", ()):
        spec = restrictable.get(str(name))
        if spec is None:
            print(f"span sweep skips {name}: not among the models run", flush=True)
            continue
        own_span = int(spec.trained_context_window) - 1
        shorter = [span for span in spans if span < own_span]
        if shorter:
            sweep[str(name)] = shorter
    return sweep


def _score_pair(
    loaded: Any,
    structured: Mapping[str, Any],
    native: Mapping[str, Any],
    span: int,
    mask_rate: float,
    context_settings: Mapping[str, Any],
    diffusion_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """One output row: the endpoints and masked accuracy of a pair at ``span``."""

    positions = masked_positions(native, mask_rate)
    with attention_span(span):
        endpoints = _final_layer_endpoints(
            loaded, structured, native, context_settings, diffusion_settings
        )
        accuracy = masked_accuracy(loaded, str(native["sequence"]), positions)
    return {
        "model": loaded.spec.name,
        "model_path": str(loaded.spec.path),
        "pair_group_id": native["pair_group_id"],
        "structured_job_id": structured["job_id"],
        "native_job_id": native["job_id"],
        "seed": native["seed"],
        "length": native["length"],
        "length_group": native["length_group"],
        "attention_span": span,
        **endpoints,
        "mask_rate": mask_rate,
        "masked_positions": len(positions),
        "mlm_accuracy": accuracy,
    }


def run(args: argparse.Namespace) -> None:
    """Evaluate every restrictable model, loading one model at a time."""

    stage_run = prepare_stage(args, "restricted_attention")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for job in stage_run.manifest_jobs:
        grouped[job["pair_group_id"]][job["control_type"]] = job
    pairs = [
        (group["structured"], group["native"])
        for group in grouped.values()
        if {"structured", "native"} <= set(group)
    ]
    if not pairs:
        raise ValueError("No complete structured/native pairs remain after filtering")
    context_settings, diffusion_settings = _metric_settings(stage_run.config)

    unrestricted: list[dict[str, str]] = []
    restrictable: dict[str, ModelSpec] = {}
    for name, spec in stage_run.runnable.items():
        applies, reason = _restriction(spec)
        if applies:
            restrictable[name] = spec
        else:
            unrestricted.append({"model": name, "reason": reason})
            print(f"unrestricted only {name}: {reason}", flush=True)

    sweep = _span_sweep(stage_run.config, restrictable)

    metadata = start_metadata(stage_run, "restricted_attention")
    metadata["selected_pairs"] = [native["pair_group_id"] for _, native in pairs]
    metadata["unrestricted_only_models"] = unrestricted
    metadata["attention_span_sweep"] = sweep
    metadata["context_config"] = context_settings
    metadata["diffusion_config"] = diffusion_settings
    metadata["definitions"] = {
        **MLM_DEFINITIONS,
        "attention_span": (
            "context_window - 1 tokens; a query attends to a key only when "
            "|query - key| <= attention_span, in every attention layer, with "
            "special tokens counted as positions; HydraRNA's state-space layers "
            "are not held"
        ),
        "attention_span_sweep": (
            f"the listed models scored again at each listed span, in "
            f"{SPAN_SWEEP_FILE}; a span that reaches every pair of a "
            "transcript changes nothing"
        ),
        "endpoints": (
            "final-layer ΔCS, C_cross, and D_distal, computed with the cosine "
            "stage's settings and sampling seeds"
        ),
        "statistical_unit": "matched structured/native pair",
    }
    metadata_path = stage_run.output_dir / "run_metadata.json"
    write_metadata(metadata_path, metadata)

    sink = CsvSink(
        stage_run.output_dir / "restricted_attention_units.csv",
        RESTRICTED_FIELDS,
        ("model", "pair_group_id"),
        resume=stage_run.resume,
    )
    sweep_sink = (
        CsvSink(
            stage_run.output_dir / SPAN_SWEEP_FILE,
            RESTRICTED_FIELDS,
            ("model", "pair_group_id", "attention_span"),
            resume=stage_run.resume,
        )
        if sweep
        else None
    )
    try:
        for model_name, spec in restrictable.items():
            own_span = int(spec.trained_context_window) - 1
            pending = [
                (structured, native)
                for structured, native in pairs
                if (model_name, native["pair_group_id"]) not in sink.completed
            ]
            sweep_pending = [
                (structured, native, span)
                for span in sweep.get(model_name, ())
                for structured, native in pairs
                if (model_name, native["pair_group_id"], str(span))
                not in sweep_sink.completed
            ]
            if not pending and not sweep_pending:
                continue
            loaded = load_encoder(
                spec,
                device=stage_run.device,
                dtype=stage_run.dtype,
                requested_max_length=stage_run.requested_max_length,
            )
            try:
                if restrict_attention(loaded) < 1:
                    raise RuntimeError(f"{model_name}: no attention layer was patched")
                for structured, native in pending:
                    sink.append(
                        [
                            _score_pair(
                                loaded,
                                structured,
                                native,
                                own_span,
                                stage_run.mask_rate,
                                context_settings,
                                diffusion_settings,
                            )
                        ]
                    )
                for structured, native, span in sweep_pending:
                    sweep_sink.append(
                        [
                            _score_pair(
                                loaded,
                                structured,
                                native,
                                span,
                                stage_run.mask_rate,
                                context_settings,
                                diffusion_settings,
                            )
                        ]
                    )
            finally:
                loaded.close()
                del loaded
                gc.collect()
            if pending:
                print(
                    f"restricted_attention {model_name}: scored {len(pending)} "
                    f"pairs with attention_span={own_span}",
                    flush=True,
                )
            if sweep_pending:
                print(
                    f"restricted_attention {model_name}: scored "
                    f"{len(sweep_pending)} pair-span rows of the span sweep "
                    f"{sweep[model_name]}",
                    flush=True,
                )
    finally:
        sink.close()
        if sweep_sink is not None:
            sweep_sink.close()

    metadata["status"] = "completed"
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_metadata(metadata_path, metadata)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    run(parse_args(argv))


if __name__ == "__main__":
    main()
