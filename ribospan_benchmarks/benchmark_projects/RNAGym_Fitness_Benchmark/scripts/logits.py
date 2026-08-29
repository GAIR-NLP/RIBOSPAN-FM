# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unified MLM log-prob interface over RNA-FM / RiNALMo / HF / HydraRNA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scripts.backends import _autocast, normalize_sequence
from scripts.hydrarna_io import encode_hydrarna_sequence


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


@dataclass
class MLMScorer:
    loaded: Any
    kind: str
    mask_id: int
    offset: int
    max_nucleotides: int

    def encode(self, sequence: str) -> torch.Tensor:
        raise NotImplementedError

    def nt_id(self, base: str) -> int:
        raise NotImplementedError

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RNAFMScorer(MLMScorer):
    def encode(self, sequence: str) -> torch.Tensor:
        rna = normalize_sequence(sequence).replace("T", "U")
        _, _, tokens = self.loaded.batch_converter([("seq", rna)])
        return tokens.to(self.loaded.device)

    def nt_id(self, base: str) -> int:
        rna = "U" if base.upper() in {"T", "U"} else base.upper()
        return int(self.loaded.alphabet.get_idx(rna))

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), _autocast(self.loaded.device, self.loaded.dtype):
            outputs = self.loaded.model(tokens)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        return _log_softmax(logits)


class RiNALMoScorer(MLMScorer):
    def encode(self, sequence: str) -> torch.Tensor:
        seq = normalize_sequence(sequence)
        return torch.tensor(
            self.loaded.alphabet.batch_tokenize([seq]),
            dtype=torch.int64,
            device=self.loaded.device,
        )

    def nt_id(self, base: str) -> int:
        # Alphabet maps U → T internally.
        return int(self.loaded.alphabet.get_idx(normalize_sequence(base)))

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), _autocast(self.loaded.device, self.loaded.dtype):
            outputs = self.loaded.model(tokens)
        return _log_softmax(outputs["logits"])


class HFScorer(MLMScorer):
    def encode(self, sequence: str) -> torch.Tensor:
        seq = normalize_sequence(sequence)
        token_ids = [self.loaded.tokenizer.token_to_id(base) for base in seq]
        return torch.tensor(
            [[self.loaded.tokenizer.cls_token_id, *token_ids, self.loaded.tokenizer.sep_token_id]],
            dtype=torch.long,
            device=self.loaded.device,
        )

    def nt_id(self, base: str) -> int:
        token_id = self.loaded.tokenizer.token_to_id(normalize_sequence(base))
        if token_id is None:
            raise KeyError(f"tokenizer has no id for {base!r}")
        return int(token_id)

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(tokens)
        kwargs = {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        with torch.inference_mode(), _autocast(self.loaded.device, self.loaded.dtype):
            try:
                outputs = self.loaded.model(**kwargs, use_cache=False)
            except TypeError:
                outputs = self.loaded.model(**kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return _log_softmax(logits)


class HydraRNAScorer(MLMScorer):
    def encode(self, sequence: str) -> torch.Tensor:
        return encode_hydrarna_sequence(sequence, self.loaded.dictionary, self.loaded.device)

    def nt_id(self, base: str) -> int:
        return int(self.loaded.dictionary.index(normalize_sequence(base)))

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        encoder = getattr(self.loaded.model, "encoder", self.loaded.model)
        with torch.inference_mode(), _autocast(self.loaded.device, self.loaded.dtype):
            outputs = encoder(tokens)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return _log_softmax(logits)


def build_scorer(loaded: Any) -> MLMScorer:
    name = type(loaded).__name__
    max_nt = int(loaded.effective_max_length)
    if name == "LoadedRNAFM":
        return RNAFMScorer(
            loaded=loaded,
            kind="rnafm",
            mask_id=int(loaded.alphabet.mask_idx),
            offset=1,
            max_nucleotides=max_nt,
        )
    if name == "LoadedRiNALMo":
        return RiNALMoScorer(
            loaded=loaded,
            kind="rinalmo",
            mask_id=int(loaded.alphabet.mask_idx),
            offset=1,
            max_nucleotides=max_nt,
        )
    if name == "LoadedHydraRNA":
        dictionary = loaded.dictionary
        mask_id = int(dictionary.index("<mask>"))
        unk = int(dictionary.unk()) if hasattr(dictionary, "unk") else -1
        if mask_id == unk:
            raise RuntimeError("HydraRNA dictionary is missing <mask>")
        return HydraRNAScorer(
            loaded=loaded,
            kind="hydrarna",
            mask_id=mask_id,
            offset=1,
            max_nucleotides=max_nt,
        )
    if name == "LoadedRNABert":
        mask_id = int(loaded.tokenizer.mask_token_id)
        return HFScorer(
            loaded=loaded,
            kind="hf",
            mask_id=mask_id,
            offset=1,
            max_nucleotides=max_nt,
        )
    raise TypeError(f"no MLM scorer for {name}")
