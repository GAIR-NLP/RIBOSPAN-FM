# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""RNAverse encoder for the frozen benchmarks.

Sequence-only: structural scores stay at the model's null value and functional
annotations stay at UNK. Rfam labels are not passed in.
"""

from __future__ import annotations

import gc
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

RIBO_BENCH = Path(__file__).resolve().parents[3]
RNAVERSE_SRC = RIBO_BENCH / "model_src" / "RNAverse"
if str(RNAVERSE_SRC) not in sys.path:
    sys.path.insert(0, str(RNAVERSE_SRC))

from rnaverse import RNAverseForMaskedLM, RNAverseTokenizer  # noqa: E402
from rnaverse.utils import tokenize_struct_score  # noqa: E402


NATIVE_NUCLEOTIDES = 1024


def normalize_sequence(sequence: str) -> str:
    """Match the RIBOSPAN benchmark: uppercase, U→T, non-ACGTN → N."""

    normalized = sequence.strip().upper().replace("U", "T")
    if not normalized:
        raise ValueError("empty sequence")
    allowed = set("ACGTN")
    if any(ch not in allowed for ch in normalized):
        normalized = "".join(ch if ch in allowed else "N" for ch in normalized)
    return normalized


def _to_rna(sequence: str) -> str:
    return normalize_sequence(sequence).replace("T", "U")


@dataclass
class LoadedRNAverse:
    spec: Any
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    config_max_length: int
    runtime_max_length: int
    effective_max_length: int
    native_nucleotides: int

    @property
    def num_hidden_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def _autocast(self):
        if self.dtype == torch.float32 or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        payload = [{"seq": _to_rna(sequence)} for sequence in sequences]
        return self._move(self.tokenizer(payload, padding=True, return_tensors="pt"))

    def _backbone(self, batch: dict[str, torch.Tensor], *, all_layers: bool):
        with torch.inference_mode(), self._autocast():
            return self.model.rnaverse(
                **batch,
                output_hidden_states=all_layers,
                output_attentions=False,
                return_dict=True,
            )

    def _nucleotide_mean(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean over nucleotide states. RNAverse uses CLS and no SEP."""

        body = hidden[:, 1:, :]
        mask = attention_mask[:, 1:].to(dtype=body.dtype).unsqueeze(-1)
        if body.shape[1] != mask.shape[1]:
            raise RuntimeError(
                f"nucleotide axis {body.shape[1]} != mask {mask.shape[1]}"
            )
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (body * mask).sum(dim=1) / denom

    @torch.inference_mode()
    def embed_pooled(
        self,
        sequence: str,
        *,
        layers: list[int],
        poolings: list[str],
    ) -> dict[int, dict[str, np.ndarray]]:
        normalized = normalize_sequence(sequence)
        if len(normalized) > self.effective_max_length:
            raise ValueError(
                f"length {len(normalized)} exceeds effective_max_length "
                f"{self.effective_max_length}"
            )
        batch = self._tokenize([normalized])
        need_all = any(int(layer) != self.num_hidden_layers for layer in layers)
        outputs = self._backbone(batch, all_layers=need_all)
        states = (
            outputs.hidden_states
            if need_all
            else (None,) * self.num_hidden_layers + (outputs.last_hidden_state,)
        )
        vectors: dict[int, dict[str, np.ndarray]] = {}
        for layer in layers:
            hidden = states[int(layer)]
            layer_vecs: dict[str, np.ndarray] = {}
            nucleotide_mean = self._nucleotide_mean(hidden, batch["attention_mask"])[0]
            cls = hidden[0, 0]
            for pooling in poolings:
                if pooling == "mean":
                    pooled = nucleotide_mean
                elif pooling == "cls":
                    pooled = cls
                elif pooling == "mean_cls":
                    pooled = 0.5 * (cls + nucleotide_mean)
                else:
                    raise ValueError(f"unsupported pooling: {pooling}")
                layer_vecs[pooling] = (
                    pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
                )
            vectors[int(layer)] = layer_vecs
        return vectors

    @torch.inference_mode()
    def embed_batch_mean(self, sequences: list[str]) -> np.ndarray:
        """Last-layer nucleotide mean, shape (batch, hidden)."""

        batch = self._tokenize(sequences)
        hidden = self._backbone(batch, all_layers=False).last_hidden_state
        pooled = self._nucleotide_mean(hidden, batch["attention_mask"])
        return pooled.detach().to(device="cpu", dtype=torch.float32).numpy()

    def hidden_states(self, sequence: str) -> tuple[torch.Tensor, ...]:
        """Embedding plus every block, nucleotide positions only, CPU float32."""

        normalized = normalize_sequence(sequence)
        if len(normalized) > self.effective_max_length:
            raise ValueError(
                f"{getattr(self.spec, 'name', 'RNAverse')}: RNA length="
                f"{len(normalized)} exceeds requested real-token maximum="
                f"{self.effective_max_length}"
            )
        batch = self._tokenize([normalized])
        outputs = self._backbone(batch, all_layers=True)
        states = outputs.hidden_states
        expected = self.num_hidden_layers + 1
        if states is None or len(states) != expected:
            actual = None if states is None else len(states)
            raise RuntimeError(f"expected {expected} hidden states, got {actual}")
        length = len(normalized)
        real = tuple(
            state[0, 1 : 1 + length].detach().to(device="cpu", dtype=torch.float32)
            for state in states
        )
        if any(state.shape[0] != length for state in real):
            raise RuntimeError("CLS removal produced an unexpected length")
        return real

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class RNAverseScorer:
    """Wild-type-background masked-marginal scorer. No SEP token; offset is 1."""

    def __init__(self, loaded: LoadedRNAverse):
        self.loaded = loaded
        self.kind = "rnaverse"
        vocab = loaded.tokenizer.dict_base_id
        self.mask_id = int(vocab["MASK"])
        self.pad_id = int(vocab["PAD"])
        self.offset = 1
        self.max_nucleotides = int(loaded.effective_max_length)

    def encode(self, sequence: str) -> torch.Tensor:
        seq = _to_rna(sequence)
        if len(seq) > self.max_nucleotides:
            raise ValueError(
                f"length {len(seq)} exceeds max_nucleotides {self.max_nucleotides}"
            )
        batch = self.loaded.tokenizer([{"seq": seq}], padding=True, return_tensors="pt")
        return batch["input_ids"].to(self.loaded.device)

    def nt_id(self, base: str) -> int:
        token = _to_rna(base)
        vocab = self.loaded.tokenizer.dict_base_id
        if token not in vocab:
            raise KeyError(f"RNAverse vocab has no id for {base!r} ({token})")
        return int(vocab[token])

    def _multimodal_from_ids(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        tokenizer = self.loaded.tokenizer
        batch, length = input_ids.shape
        device = input_ids.device
        attention = input_ids.ne(self.pad_id).to(dtype=torch.float32)
        null = float(tokenizer.struct_score_null_value)
        # Binning builds its edges on CPU; keep the scores there, then move ids.
        struct_scores = torch.full((batch, length - 1), null)
        struct_tokens = tokenize_struct_score(
            struct_scores,
            bin_num=int(tokenizer.struct_score_bin_num),
            null_value=tokenizer.struct_score_null_value,
        ).to(device)
        probe_id = int(tokenizer.dict_probe_method_id["UNK"])
        probe = torch.full(
            (batch, length - 1), probe_id, dtype=torch.long, device=device
        )
        func_pad = int(tokenizer.func_token_pad_len)
        func = torch.full(
            (batch, func_pad),
            int(tokenizer.dict_func_token["PAD"]),
            dtype=torch.long,
            device=device,
        )
        func[:, 0] = int(tokenizer.dict_func_token["UNK"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "input_struct_tokens": struct_tokens,
            "input_probe_method_tokens": probe,
            "input_func_tokens": func,
        }

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = self._multimodal_from_ids(tokens)
        with torch.inference_mode(), self.loaded._autocast():
            outputs = self.loaded.model(**batch, return_dict=True)
        return torch.log_softmax(outputs.logits.float(), dim=-1)


def find_checkpoint(root: Path | None = None) -> Path | None:
    base = root or RIBO_BENCH
    candidates = [
        base / "model_weights" / "rnaverse",
        base / "model_weights" / "RNAverse",
        base / "RNAverse" / "model_weights" / "rnaverse",
        base.parent
        / "RIBOSPAN-FM"
        / "ribospan_benchmarks"
        / "model_weights"
        / "RNAverse",
    ]
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    weights = base / "model_weights"
    if weights.is_dir():
        for config in weights.glob("*/config.json"):
            return config.parent
    return None


def load_rnaverse_encoder(
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedRNAverse:
    path = Path(spec.path)
    tokenizer = RNAverseTokenizer.from_pretrained(str(path))
    model = RNAverseForMaskedLM.from_pretrained(str(path))
    model.eval()
    if dtype != torch.float32:
        model.to(dtype=dtype)
    model.to(device)

    config_max = int(model.config.max_position_embeddings)
    native = config_max - 1
    if native < 1:
        raise RuntimeError(f"unexpected max_position_embeddings={config_max}")
    if requested_max_length is None:
        effective = native
    else:
        effective = int(requested_max_length)
    vocab = tokenizer.dict_base_id
    missing = [token for token in ("A", "C", "G", "U", "N", "MASK", "CLS", "PAD") if token not in vocab]
    if missing:
        raise RuntimeError(
            f"RNAverse seq vocab is missing {missing}; keys={sorted(vocab)}"
        )
    return LoadedRNAverse(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        config_max_length=config_max,
        runtime_max_length=effective + 1,
        effective_max_length=effective,
        native_nucleotides=native,
    )
