# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""RiboGRAM encoder for the frozen benchmarks.

Sequence only. Mean pooling drops CLS and EOS. The benchmark window is 1024 nt.
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
RIBOGRAM_SRC = RIBO_BENCH / "model_src" / "RiboGRAM"
if str(RIBOGRAM_SRC) not in sys.path:
    sys.path.insert(0, str(RIBOGRAM_SRC))

from ribogram import load_model_and_alphabet  # noqa: E402


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


@dataclass
class LoadedRiboGRAM:
    spec: Any
    model: Any
    alphabet: Any
    converter: Any
    device: torch.device
    dtype: torch.dtype
    config_max_length: int
    runtime_max_length: int
    effective_max_length: int
    native_nucleotides: int

    @property
    def num_hidden_layers(self) -> int:
        return int(self.model.num_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.embed_dim)

    def _autocast(self):
        if self.dtype == torch.float32 or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def _tokens(self, sequences: list[str]) -> torch.Tensor:
        records = [(index, sequence) for index, sequence in enumerate(sequences)]
        _labels, _seqs, tokens = self.converter(records)
        return tokens.to(self.device)

    def _biological(self, tokens: torch.Tensor) -> torch.Tensor:
        alphabet = self.alphabet
        return (
            tokens.ne(alphabet.padding_idx)
            & tokens.ne(alphabet.cls_idx)
            & tokens.ne(alphabet.eos_idx)
        )

    @torch.inference_mode()
    def embed_batch_mean(self, sequences: list[str]) -> np.ndarray:
        """Last-layer nucleotide mean, shape (batch, hidden)."""

        tokens = self._tokens(sequences)
        layer = self.num_hidden_layers
        with self._autocast():
            hidden = self.model(tokens, repr_layers=[layer])["hidden_representations"][
                layer
            ]
        mask = self._biological(tokens).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (hidden.float() * mask).sum(dim=1) / denom
        return pooled.detach().to(device="cpu", dtype=torch.float32).numpy()

    def hidden_states(self, sequence: str) -> tuple[torch.Tensor, ...]:
        """Embedding plus every block, nucleotide positions only, CPU float32."""

        normalized = normalize_sequence(sequence)
        if len(normalized) > self.effective_max_length:
            raise ValueError(
                f"{getattr(self.spec, 'name', 'RiboGRAM')}: RNA length="
                f"{len(normalized)} exceeds requested real-token maximum="
                f"{self.effective_max_length}"
            )
        tokens = self._tokens([normalized])
        layers = list(range(self.num_hidden_layers + 1))
        with torch.inference_mode(), self._autocast():
            hidden = self.model(tokens, repr_layers=layers)["hidden_representations"]
        keep = self._biological(tokens)[0]
        real = []
        for layer in layers:
            state = hidden[layer][0, keep].detach().to(device="cpu", dtype=torch.float32)
            if state.shape[0] != len(normalized):
                raise RuntimeError(
                    f"layer {layer} nucleotide count {state.shape[0]} != {len(normalized)}"
                )
            real.append(state)
        return tuple(real)

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class RiboGRAMScorer:
    """Wild-type-background masked-marginal scorer. CLS offset is 1; EOS follows."""

    def __init__(self, loaded: LoadedRiboGRAM):
        self.loaded = loaded
        self.kind = "ribogram"
        self.mask_id = int(loaded.alphabet.mask_idx)
        self.offset = 1
        self.max_nucleotides = int(loaded.effective_max_length)

    def encode(self, sequence: str) -> torch.Tensor:
        seq = normalize_sequence(sequence)
        if len(seq) > self.max_nucleotides:
            raise ValueError(
                f"length {len(seq)} exceeds max_nucleotides {self.max_nucleotides}"
            )
        return self.loaded._tokens([seq])

    def nt_id(self, base: str) -> int:
        token = normalize_sequence(base).replace("T", "U")
        if token not in self.loaded.alphabet.tok_to_idx:
            raise KeyError(f"RiboGRAM vocab has no id for {base!r}")
        return int(self.loaded.alphabet.get_idx(token))

    def log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), self.loaded._autocast():
            logits = self.loaded.model(tokens)["logits"]
        return torch.log_softmax(logits.float(), dim=-1)


def find_checkpoint(root: Path | None = None) -> Path | None:
    base = root or RIBO_BENCH
    candidates = [
        base / "model_weights" / "RiboGRAM" / "RiboGRAM.pkl",
        base / "RiboGRAM" / "checkpoints" / "RiboGRAM.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_ribogram_encoder(
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedRiboGRAM:
    path = Path(spec.path)
    _alphabet, converter, model = load_model_and_alphabet(path, architecture="RiboGRAM")
    model.eval()
    if dtype != torch.float32:
        model.to(dtype=dtype)
    model.to(device)
    native = NATIVE_NUCLEOTIDES
    effective = native if requested_max_length is None else int(requested_max_length)
    if effective < 1:
        raise RuntimeError(f"invalid requested length {effective}")
    return LoadedRiboGRAM(
        spec=spec,
        model=model,
        alphabet=_alphabet,
        converter=converter,
        device=device,
        dtype=dtype,
        config_max_length=native + 2,
        runtime_max_length=effective + 2,
        effective_max_length=effective,
        native_nucleotides=native,
    )
