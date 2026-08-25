# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""HydraRNA hidden-state backend for the cosine benchmark."""

from __future__ import annotations

import gc
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model_io import ModelSpec, normalize_sequence
from .model_src import HYDRARNA_SRC_ROOT


CHECKPOINT_CANDIDATES = (
    "HydraRNA_model.pt",
    "hydrarna.pt",
    "checkpoint_best.pt",
)


@dataclass
class LoadedHydraRNA:
    """Fairseq HydraRNA encoder with the cosine ``hidden_states`` surface."""

    spec: ModelSpec
    model: Any
    dictionary: Any
    device: torch.device
    dtype: torch.dtype
    config_max_length: int
    runtime_max_length: int
    effective_max_length: int
    rope_scaling: dict[str, Any] | None = None

    @property
    def num_hidden_layers(self) -> int:
        return int(len(self.model.encoder.backbone.layers))

    def hidden_states(self, sequence: str) -> tuple[torch.Tensor, ...]:
        normalized = normalize_sequence(sequence)
        if len(normalized) > self.effective_max_length:
            raise ValueError(
                f"{self.spec.name}: RNA length={len(normalized)} exceeds "
                f"requested real-token maximum={self.effective_max_length}"
            )
        src_tokens = encode_hydrarna_sequence(
            normalized, self.dictionary, self.device
        )
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.dtype != torch.float32
            else nullcontext()
        )
        encoder = self.model.encoder
        with torch.inference_mode(), autocast_context:
            hidden = _encoder_hidden_states(encoder, src_tokens)
        real_hidden = tuple(
            state[0, 1:-1].detach().to(device="cpu", dtype=torch.float32)
            for state in hidden
        )
        if any(state.shape[0] != len(normalized) for state in real_hidden):
            lengths = [int(state.shape[0]) for state in real_hidden]
            raise RuntimeError(
                "HydraRNA special-token removal produced unexpected lengths "
                f"{lengths} for RNA length {len(normalized)}"
            )
        del src_tokens, hidden
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return real_hidden

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def format_hydrarna_line(sequence: str) -> str:
    """Space-separated nucleotides with a leading ``<s>``, matching HydraRNA."""

    normalized = normalize_sequence(sequence)
    return "<s> " + " ".join(normalized)


def encode_hydrarna_sequence(
    sequence: str, dictionary: Any, device: torch.device
) -> torch.Tensor:
    tokens = dictionary.encode_line(
        format_hydrarna_line(sequence),
        add_if_not_exist=False,
        append_eos=True,
    ).long()
    return tokens.unsqueeze(0).to(device)


def resolve_hydrarna_root(spec: ModelSpec | None = None) -> Path:
    """Locate the HydraRNA/fairseq checkout."""

    candidates: list[Path] = []
    env = os.environ.get("HYDRARNA_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(HYDRARNA_SRC_ROOT)
    if spec is not None:
        candidates.append(spec.path / "src")
    for candidate in candidates:
        fairseq_root = candidate / "fairseq"
        if fairseq_root.is_dir() and (fairseq_root / "fairseq").is_dir():
            return candidate.resolve()
        if (candidate / "fairseq").is_dir() and (candidate / "fairseq" / "models").is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "HydraRNA source checkout not found. Clone "
        "https://github.com/GuipengLi/HydraRNA and set HYDRARNA_ROOT, "
        f"or place it at ribospan_benchmarks/model_src/HydraRNA. Looked in: {searched}"
    )


def resolve_hydrarna_checkpoint(spec: ModelSpec) -> Path:
    for name in CHECKPOINT_CANDIDATES:
        path = spec.path / name
        if path.is_file():
            return path
    found = sorted(spec.path.glob("*.pt"))
    if len(found) == 1:
        return found[0]
    if found:
        names = ", ".join(path.name for path in found)
        raise FileNotFoundError(
            f"{spec.name}: multiple checkpoints in {spec.path} ({names}); "
            "keep HydraRNA_model.pt"
        )
    raise FileNotFoundError(
        f"{spec.name}: no HydraRNA .pt checkpoint in {spec.path}. "
        "Download HydraRNA_model.pt from the official Google Drive / Zenodo."
    )


def resolve_hydrarna_dict_dir(spec: ModelSpec, source_root: Path) -> Path:
    for candidate in (spec.path / "dict", source_root / "dict"):
        if candidate.is_dir() and (candidate / "dict.txt").is_file():
            return candidate
    raise FileNotFoundError(
        f"{spec.name}: dict/dict.txt not found under {spec.path} or {source_root}"
    )


def _patch_dataclass_mutable_defaults() -> None:
    """Fairseq 0.12 still uses mutable dataclass defaults (Python 3.10 style)."""

    import dataclasses

    if getattr(dataclasses, "_hydrarna_mutable_defaults", False):
        return
    original = dataclasses._get_field

    def _get_field(cls, a_name, a_type, default_kw_only):
        try:
            return original(cls, a_name, a_type, default_kw_only)
        except ValueError as exc:
            if "mutable default" not in str(exc):
                raise
            current = cls.__dict__.get(a_name, dataclasses.MISSING)
            if isinstance(current, dataclasses.Field):
                value = current.default
                current.default = dataclasses.MISSING
                current.default_factory = lambda cached=value: cached
            else:
                setattr(
                    cls,
                    a_name,
                    dataclasses.field(default_factory=lambda cached=current: cached),
                )
            return original(cls, a_name, a_type, default_kw_only)

    dataclasses._get_field = _get_field
    dataclasses._hydrarna_mutable_defaults = True


def _patch_torch_load_for_fairseq() -> None:
    """Fairseq checkpoints store argparse.Namespace; torch 2.6 defaults to weights_only."""

    if getattr(torch, "_hydrarna_load_patched", False):
        return
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    torch._hydrarna_load_patched = True


def _prepend_fairseq(source_root: Path) -> None:
    fairseq_root = source_root / "fairseq"
    if not (fairseq_root / "fairseq").is_dir():
        raise FileNotFoundError(f"fairseq package missing under {fairseq_root}")
    _patch_dataclass_mutable_defaults()
    _patch_torch_load_for_fairseq()
    path = str(fairseq_root.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _encoder_hidden_states(encoder: Any, src_tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Walk mixers to recover embedding + per-layer states.

    The last tensor matches official ``extract_features`` (post ``norm_f``).
    """

    x, _embed = encoder.forward_embedding(src_tokens)
    encoder_padding_mask = src_tokens.eq(encoder.padding_idx)
    has_pads = encoder_padding_mask.any()
    keep = 1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
    states = [x]
    layers = encoder.backbone.layers
    for index, layer in enumerate(layers):
        y = layer.mixer(layer.norm(x))
        x = (y + x) * keep
        if index == len(layers) - 1:
            x = encoder.backbone.norm_f(x)
        states.append(x)
    return tuple(states)


def load_hydrarna(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedHydraRNA:
    """Load HydraRNA through the official fairseq masked-LM task."""

    source_root = resolve_hydrarna_root(spec)
    _prepend_fairseq(source_root)
    try:
        import fairseq.models.hydraAttRNA  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HydraRNA architecture is not importable. Use a full clone of "
            "https://github.com/GuipengLi/HydraRNA and install that fairseq "
            "environment before loading this backend."
        ) from exc
    checkpoint = resolve_hydrarna_checkpoint(spec)
    dict_dir = resolve_hydrarna_dict_dir(spec, source_root)

    from fairseq import checkpoint_utils, options, tasks

    parser = options.get_generation_parser(default_task="masked_lm_span")
    args = options.parse_args_and_arch(parser, [str(dict_dir)])
    task = tasks.setup_task(args)
    models, _saved_args = checkpoint_utils.load_model_ensemble(
        [str(checkpoint)], task=task
    )
    model = models[0]
    model.eval()
    model.to(device)
    if dtype == torch.float16:
        model.half()
    elif dtype == torch.bfloat16:
        model.to(dtype=torch.bfloat16)

    requested_real_tokens = (
        requested_max_length
        if requested_max_length is not None
        else spec.trained_context_window
    )
    encoder = model.encoder
    config_max = int(getattr(getattr(encoder, "args", None), "max_positions", 0) or 0)
    if config_max < 1:
        config_max = int(encoder.max_positions())
    runtime_max = max(config_max, int(requested_real_tokens) + 2)
    setattr(model, "config", type("HydraRNAConfig", (), {
        "position_embedding_type": "hydra",
        "max_position_embeddings": runtime_max,
        "num_hidden_layers": int(len(encoder.backbone.layers)),
    })())
    return LoadedHydraRNA(
        spec=spec,
        model=model,
        dictionary=task.source_dictionary,
        device=device,
        dtype=dtype,
        config_max_length=config_max,
        runtime_max_length=runtime_max,
        effective_max_length=int(requested_real_tokens),
        rope_scaling=None,
    )
