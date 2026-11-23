# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""LIMBER encoder for frozen mRNABench probes."""

from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_CLS = 2
_SEP = 3
_NUC = {"A": 5, "U": 6, "C": 7, "G": 8, "N": 9}
_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
    "type_vocab_size",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "pad_token_id",
    "layer_pattern",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "tie_word_embeddings",
)


def _limber_src() -> Path:
    return Path(__file__).resolve().parents[3] / "model_src" / "LIMBER" / "models"


def _to_limber_alphabet(sequence: str) -> str:
    from scripts.backends import normalize_sequence

    return normalize_sequence(sequence).replace("T", "U")


@dataclass
class LoadedLIMBER:
    spec: Any
    model: Any
    device: torch.device
    dtype: torch.dtype
    effective_max_length: int
    hidden_size: int
    num_hidden_layers: int

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _autocast(self):
        if self.dtype == torch.float32 or self.device.type != "cuda":
            from contextlib import nullcontext

            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def _last_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        model = self.model
        hidden = model.embeddings(input_ids=input_ids, token_type_ids=None)
        ext_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden.dtype))
        ext_mask = ext_mask * torch.finfo(hidden.dtype).min
        for layer, is_mamba in zip(model.layers, model.is_mamba):
            if is_mamba:
                hidden = layer(hidden)
            else:
                hidden = layer(hidden, attention_mask=ext_mask)[0]
        return hidden

    @torch.inference_mode()
    def embed_pooled(
        self,
        sequence: str,
        *,
        layers: list[int],
        poolings: list[str],
    ) -> dict[int, dict[str, np.ndarray]]:
        seq = _to_limber_alphabet(sequence)
        if len(seq) > self.effective_max_length:
            raise ValueError(
                f"length {len(seq)} exceeds effective_max_length {self.effective_max_length}"
            )
        if any(int(layer) != self.num_hidden_layers for layer in layers):
            raise ValueError("LIMBER mRNABench embeddings use the final layer only")
        token_ids = [_CLS, *(_NUC[base] for base in seq), _SEP]
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(tokens)
        with self._autocast():
            hidden = self._last_hidden(tokens, attention)[0]
        nucleotide_mean = hidden[1:-1].mean(dim=0)
        cls = hidden[0]
        layer_vecs: dict[str, np.ndarray] = {}
        for pooling in poolings:
            if pooling == "mean":
                pooled = nucleotide_mean
            elif pooling == "cls":
                pooled = cls
            elif pooling == "mean_cls":
                pooled = 0.5 * (cls + nucleotide_mean)
            else:
                raise ValueError(f"unsupported pooling: {pooling}")
            layer_vecs[pooling] = pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
        return {self.num_hidden_layers: layer_vecs}


def load_limber_encoder(
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedLIMBER:
    src = _limber_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM

    weights = Path(spec.path)
    config_path = weights / "model_config.json"
    weight_path = weights / "pytorch_model.bin"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"LIMBER checkpoint is incomplete under {weights}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = HybridMambaConfig(**{key: payload[key] for key in _CONFIG_KEYS if key in payload})
    native = int(config.max_position_embeddings) - 2
    requested = int(
        requested_max_length if requested_max_length is not None else spec.trained_context_window
    )
    if requested > native:
        raise ValueError(
            f"{spec.name}: context_window {requested} exceeds position table capacity {native}"
        )
    model = HybridMambaForMaskedLM(config)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"LIMBER state mismatch missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}"
        )
    model.to(device=device, dtype=dtype)
    model.eval()
    return LoadedLIMBER(
        spec=spec,
        model=model,
        device=device,
        dtype=dtype,
        effective_max_length=requested,
        hidden_size=int(config.hidden_size),
        num_hidden_layers=int(config.num_hidden_layers),
    )
