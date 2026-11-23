# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""LIMBER-macro masked LM for RNAGym masked-marginal scoring."""

from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.backends import _autocast, normalize_sequence

# BERT-style kmer1 vocab used by the LIMBER checkpoint.
_CLS = 2
_SEP = 3
_MASK = 4
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


def _nucleotide_id(base: str) -> int:
    token = normalize_sequence(base).replace("T", "U")
    if token not in _NUC:
        raise KeyError(f"LIMBER vocab has no id for {base!r}")
    return _NUC[token]


@dataclass
class LoadedLIMBER:
    spec: Any
    model: Any
    device: torch.device
    dtype: torch.dtype
    effective_max_length: int

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class LIMBERScorer:
    """Wild-type-background masked-marginal scorer. CLS offset is 1; SEP follows."""

    def __init__(self, loaded: LoadedLIMBER):
        self.loaded = loaded
        self.kind = "limber"
        self.mask_id = _MASK
        self.offset = 1
        self.max_nucleotides = int(loaded.effective_max_length)

    def encode(self, sequence: str) -> torch.Tensor:
        seq = normalize_sequence(sequence)
        if len(seq) > self.max_nucleotides:
            raise ValueError(
                f"length {len(seq)} exceeds max_nucleotides {self.max_nucleotides}"
            )
        token_ids = [_CLS, *(_nucleotide_id(base) for base in seq), _SEP]
        return torch.tensor([token_ids], dtype=torch.long, device=self.loaded.device)

    def nt_id(self, base: str) -> int:
        return _nucleotide_id(base)

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        attention = torch.ones_like(tokens)
        with torch.inference_mode(), _autocast(self.loaded.device, self.loaded.dtype):
            logits = self.loaded.model(
                input_ids=tokens,
                attention_mask=attention,
            ).logits
        return torch.log_softmax(logits.float(), dim=-1)


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
    max_nucleotides = int(config.max_position_embeddings) - 2
    requested = int(requested_max_length if requested_max_length is not None else spec.trained_context_window)
    if requested > max_nucleotides:
        raise ValueError(
            f"{spec.name}: context_window {requested} exceeds "
            f"position table capacity {max_nucleotides}"
        )
    model = HybridMambaForMaskedLM(config)
    state = torch.load(weight_path, map_location="cpu")
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
    )
