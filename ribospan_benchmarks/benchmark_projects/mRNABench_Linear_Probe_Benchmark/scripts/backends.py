# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Encoder backends (RiNALMo / RNA-FM / HydraRNA) with a shared pooling contract."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.model_io import ModelSpec
from scripts.model_src import ensure_import_path


def normalize_sequence(sequence: str) -> str:
    """Uppercase, U→T; map non-ACGTN bases to N."""

    normalized = sequence.strip().upper().replace("U", "T")
    if not normalized:
        raise ValueError("empty sequence")
    allowed = {"A", "C", "G", "T", "N"}
    if any(ch not in allowed for ch in normalized):
        normalized = "".join(ch if ch in allowed else "N" for ch in normalized)
    return normalized


def _pool_cls_body(hidden: torch.Tensor, *, pooling: str) -> torch.Tensor:
    """Pool ``[CLS] + body + [EOS/SEP]`` token states (seq-first)."""
    if pooling == "mean":
        return hidden[1:-1].mean(dim=0)
    if pooling == "cls":
        return hidden[0]
    if pooling == "mean_cls":
        return 0.5 * (hidden[0] + hidden[1:-1].mean(dim=0))
    raise ValueError(f"unsupported pooling: {pooling}")


def _autocast(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@dataclass
class LoadedRiNALMo:
    spec: ModelSpec
    model: Any
    alphabet: Any
    device: torch.device
    dtype: torch.dtype
    effective_max_length: int
    num_hidden_layers: int
    hidden_size: int

    @torch.inference_mode()
    def embed_pooled(
        self,
        sequence: str,
        *,
        layers: list[int],
        poolings: list[str],
    ) -> dict[int, dict[str, np.ndarray]]:
        # The public API only exposes the final block representation.
        last_layer = self.num_hidden_layers
        bad = [layer for layer in layers if layer != last_layer]
        if bad:
            raise ValueError(
                f"RiNALMo backend only supports the final layer ({last_layer}); "
                f"got {layers}"
            )

        normalized = normalize_sequence(sequence)
        if len(normalized) > self.effective_max_length:
            raise ValueError(
                f"length {len(normalized)} exceeds effective_max_length "
                f"{self.effective_max_length}"
            )

        tokens = torch.tensor(
            self.alphabet.batch_tokenize([normalized]),
            dtype=torch.int64,
            device=self.device,
        )
        with _autocast(self.device, self.dtype):
            outputs = self.model(tokens)
        # [1, L, D] -> [L, D]; layout is <cls> + bases + <eos>
        hidden = outputs["representation"][0]
        layer_vecs: dict[str, np.ndarray] = {}
        for pooling in poolings:
            pooled = _pool_cls_body(hidden, pooling=pooling)
            layer_vecs[pooling] = pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
        return {last_layer: layer_vecs}

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass
class LoadedRNAFM:
    spec: ModelSpec
    model: Any
    alphabet: Any
    batch_converter: Any
    device: torch.device
    dtype: torch.dtype
    effective_max_length: int
    num_hidden_layers: int
    hidden_size: int

    @torch.inference_mode()
    def embed_pooled(
        self,
        sequence: str,
        *,
        layers: list[int],
        poolings: list[str],
    ) -> dict[int, dict[str, np.ndarray]]:
        normalized = normalize_sequence(sequence)
        # RNA-FM alphabet expects U (T is not in vocab).
        rna_seq = normalized.replace("T", "U")
        if len(rna_seq) > self.effective_max_length:
            raise ValueError(
                f"length {len(rna_seq)} exceeds effective_max_length "
                f"{self.effective_max_length}"
            )

        _, _, tokens = self.batch_converter([("seq", rna_seq)])
        tokens = tokens.to(self.device)
        with _autocast(self.device, self.dtype):
            results = self.model(tokens, repr_layers=layers, need_head_weights=False)

        vectors: dict[int, dict[str, np.ndarray]] = {}
        for layer in layers:
            # ESM-style: <cls> + bases + <eos>
            hidden = results["representations"][layer][0]
            layer_vecs: dict[str, np.ndarray] = {}
            for pooling in poolings:
                pooled = _pool_cls_body(hidden, pooling=pooling)
                layer_vecs[pooling] = (
                    pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
                )
            vectors[layer] = layer_vecs
        return vectors

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _rinalmo_flash_to_eager_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Rewrite Flash fused ``Wqkv`` keys to eager ``to_q`` / ``to_k`` / ``to_v``."""
    converted: dict[str, Any] = {}
    for key, value in state.items():
        if key.endswith(".mh_attn.Wqkv.weight"):
            prefix = key[: -len("Wqkv.weight")]
            # Flash stores fused [3D, D]; eager uses separate to_q/to_k/to_v.
            q_w, k_w, v_w = value.chunk(3, dim=0)
            converted[prefix + "mh_attn.to_q.weight"] = q_w.contiguous()
            converted[prefix + "mh_attn.to_k.weight"] = k_w.contiguous()
            converted[prefix + "mh_attn.to_v.weight"] = v_w.contiguous()
        elif key.endswith(".mh_attn.Wqkv.bias"):
            prefix = key[: -len("Wqkv.bias")]
            q_b, k_b, v_b = value.chunk(3, dim=0)
            converted[prefix + "mh_attn.to_q.bias"] = q_b.contiguous()
            converted[prefix + "mh_attn.to_k.bias"] = k_b.contiguous()
            converted[prefix + "mh_attn.to_v.bias"] = v_b.contiguous()
        elif ".mh_attn.out_proj." in key:
            converted[key.replace(".mh_attn.out_proj.", ".mh_attn.mh_attn.out_proj.")] = value
        else:
            converted[key] = value
    return converted


def load_rinalmo(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedRiNALMo:
    ensure_import_path()
    from rinalmo.config import model_config
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.model.model import RiNALMo

    weight_path = Path(spec.path)
    if weight_path.is_dir():
        candidates = sorted(weight_path.glob("*.pt")) + sorted(weight_path.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No .pt/.pth weights under {weight_path}")
        weight_path = candidates[0]
    if not weight_path.is_file():
        raise FileNotFoundError(f"RiNALMo weight file not found: {weight_path}")

    config = model_config("giga")
    # Prefer the PyTorch attention path so flash-attn is optional.
    config.model.transformer.use_flash_attn = False
    model = RiNALMo(config)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if any(k.endswith(".mh_attn.Wqkv.weight") for k in state):
        state = _rinalmo_flash_to_eager_state_dict(state)
    # rotary_emb.inv_freq is a non-persistent buffer recomputed at init.
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rotary_emb.inv_freq" not in k]
    if missing or unexpected:
        raise RuntimeError(
            "RiNALMo state_dict mismatch after flash→eager conversion: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    model.eval()
    model.to(device)

    alphabet = Alphabet(**config["alphabet"])
    effective = int(
        requested_max_length
        if requested_max_length is not None
        else spec.trained_context_window
    )
    return LoadedRiNALMo(
        spec=spec,
        model=model,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
        effective_max_length=effective,
        num_hidden_layers=int(config.model.transformer.num_blocks),
        hidden_size=int(config.globals.embed_dim),
    )


def load_rnafm(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedRNAFM:
    import fm
    from fm.pretrained import load_model_and_alphabet_core

    weight_path = Path(spec.path)
    if weight_path.is_dir():
        preferred = weight_path / "RNA-FM_pretrained.pth"
        weight_path = preferred if preferred.is_file() else next(
            iter(sorted(weight_path.glob("*.pth")) + sorted(weight_path.glob("*.pt"))),
            None,
        )
        if weight_path is None:
            raise FileNotFoundError(f"No .pth/.pt weights under {spec.path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"RNA-FM weight file not found: {weight_path}")

    # PyTorch>=2.6 defaults weights_only=True; RNA-FM checkpoints pickle Namespace.
    model_data = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    model, alphabet = load_model_and_alphabet_core(
        "rna-fm", model_data, regression_data=None, theme="rna"
    )
    model.eval()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()

    # RNA-FM max positions are typically 1024 tokens including specials.
    # Keep one slot each for <cls>/<eos>.
    declared = int(
        requested_max_length
        if requested_max_length is not None
        else spec.trained_context_window
    )
    max_positions = int(getattr(model.args, "max_positions", declared + 2))
    effective = min(declared, max_positions - 2)

    return LoadedRNAFM(
        spec=spec,
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
        device=device,
        dtype=dtype,
        effective_max_length=effective,
        num_hidden_layers=int(model.args.layers),
        hidden_size=int(model.args.embed_dim),
    )


def hydrarna_runtime_available() -> tuple[bool, str]:
    """Return whether this process can load the HydraRNA/fairseq stack."""

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False, (
            "mamba_ssm is not importable; extract HydraRNA in the benchmark "
            "conda environment"
        )
    return True, ""


def backend_runtime_available(backend: str | None) -> tuple[bool, str]:
    name = (backend or "rnabert").lower()
    if name in {"hydrarna", "hydra-rna", "hydra"}:
        return hydrarna_runtime_available()
    return True, ""


def load_encoder(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
):
    """Dispatch to the backend declared in ``ModelSpec.backend``."""
    backend = (spec.backend or "rnabert").lower()
    if backend in {"rnabert", "ribospan"}:
        from scripts.model_io import load_hf_masked_lm

        return load_hf_masked_lm(
            spec,
            device=device,
            dtype=dtype,
            requested_max_length=requested_max_length,
            backend=backend,
        )
    if backend in {"rinalmo", "rinalmo-giga"}:
        return load_rinalmo(
            spec,
            device=device,
            dtype=dtype,
            requested_max_length=requested_max_length,
        )
    if backend in {"rnafm", "rna-fm"}:
        return load_rnafm(
            spec,
            device=device,
            dtype=dtype,
            requested_max_length=requested_max_length,
        )
    if backend in {"hydrarna", "hydra-rna", "hydra"}:
        from scripts.hydrarna_io import load_hydrarna

        return load_hydrarna(
            spec,
            device=device,
            dtype=dtype,
            requested_max_length=requested_max_length,
        )
    raise ValueError(f"unsupported model backend: {backend!r}")
