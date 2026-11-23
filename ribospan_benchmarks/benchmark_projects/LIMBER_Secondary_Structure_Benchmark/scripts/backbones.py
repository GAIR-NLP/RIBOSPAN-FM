# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Encoder adapters. The secondary-structure head and schedule stay the same."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIMBER_SRC = PROJECT_ROOT.parents[1] / "model_src" / "LIMBER" / "models"
RIBOSPAN_SRC = PROJECT_ROOT.parents[1] / "model_src"

_LIMBER_KEYS = (
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
_LIMBER_NUC = {"A": 5, "U": 6, "C": 7, "G": 8, "N": 9}
_LIMBER_CLS = 2
_LIMBER_SEP = 3


class StructureBackbone:
    """A frozen-then-gradually-unfrozen encoder with one block per entry in ``layers``."""

    def __init__(self, name: str, module: nn.Module, layers: nn.ModuleList, hidden_size: int):
        self.name = name
        self.module = module
        self.layers = layers
        self.hidden_size = hidden_size

    def hidden(self, sequence: str, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class LimberBackbone(StructureBackbone):
    def hidden(self, sequence: str, device: torch.device) -> torch.Tensor:
        token_ids = [_LIMBER_CLS] + [_LIMBER_NUC[base] for base in sequence] + [_LIMBER_SEP]
        input_ids = torch.tensor([token_ids], device=device)
        attention = torch.ones_like(input_ids)
        states = self.module.embeddings(input_ids=input_ids, token_type_ids=None)
        extended = (1.0 - attention[:, None, None, :].to(states.dtype)) * torch.finfo(states.dtype).min
        for layer, is_mamba in zip(self.module.layers, self.module.is_mamba):
            if is_mamba:
                states = layer(states)
            else:
                states = layer(states, attention_mask=extended)[0]
        return states[:, 1:1 + len(sequence)]


class RibospanBackbone(StructureBackbone):
    def __init__(self, name: str, module: nn.Module, layers: nn.ModuleList, hidden_size: int, tokenizer):
        super().__init__(name, module, layers, hidden_size)
        self.tokenizer = tokenizer

    def hidden(self, sequence: str, device: torch.device) -> torch.Tensor:
        translated = sequence.replace("U", "T")
        unknown = self.tokenizer.unk_token_id
        bases = []
        for base in translated:
            token = self.tokenizer.token_to_id(base)
            bases.append(unknown if token is None else token)
        token_ids = [self.tokenizer.cls_token_id, *bases, self.tokenizer.sep_token_id]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)
        outputs = self.module.ribospan(
            input_ids=input_ids,
            attention_mask=attention,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, 1:1 + len(sequence)]


def _load_limber(weights: Path, device: torch.device) -> LimberBackbone:
    if str(LIMBER_SRC) not in sys.path:
        sys.path.insert(0, str(LIMBER_SRC))
    from hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM

    config_path = weights / "model_config.json"
    weight_path = weights / "pytorch_model.bin"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"LIMBER-macro checkpoint is incomplete under {weights}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = HybridMambaConfig(**{key: payload[key] for key in _LIMBER_KEYS if key in payload})
    model = HybridMambaForMaskedLM(config)
    state = torch.load(weight_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"LIMBER state mismatch missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    model.to(device)
    return LimberBackbone("LIMBER-macro", model, model.layers, int(config.hidden_size))


def _load_ribospan(name: str, weights: Path, device: torch.device) -> RibospanBackbone:
    if str(RIBOSPAN_SRC) not in sys.path:
        sys.path.insert(0, str(RIBOSPAN_SRC))
    from ribospan import RiboSpanConfig, RiboSpanForMaskedLM, RiboSpanTokenizer

    vocab = Path(sys.modules["ribospan"].__file__).resolve().parent / "vocab.txt"
    tokenizer = RiboSpanTokenizer(str(vocab))
    config = RiboSpanConfig.from_pretrained(str(weights), local_files_only=True)
    model = RiboSpanForMaskedLM.from_pretrained(
        str(weights),
        config=config,
        local_files_only=True,
    )
    model.to(device)
    return RibospanBackbone(
        name,
        model,
        model.ribospan.encoder.layer,
        int(config.hidden_size),
        tokenizer,
    )


def load_backbone(row: dict, device: torch.device) -> StructureBackbone:
    backend = row["backend"]
    weights = Path(row["weights"])
    if backend == "limber":
        return _load_limber(weights, device)
    if backend == "ribospan":
        return _load_ribospan(str(row["name"]), weights, device)
    raise ValueError(f"unknown backend {backend}")
