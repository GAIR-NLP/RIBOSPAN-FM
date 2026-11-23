# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Frozen RIBOSPAN last-layer states, CLS plus nucleotides, no SEP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

MODEL_SRC = Path(__file__).resolve().parents[3] / "model_src"


def _import_ribospan():
    root = str(MODEL_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)
    from ribospan import RiboSpanConfig, RiboSpanForMaskedLM, RiboSpanTokenizer

    vocab = Path(sys.modules["ribospan"].__file__).resolve().parent / "vocab.txt"
    return RiboSpanConfig, RiboSpanForMaskedLM, RiboSpanTokenizer, vocab


def normalize_sequence(sequence: str) -> str:
    """Map U to T. Non-ACGT symbols, including IUPAC codes, become N."""

    normalized = sequence.strip().upper().replace("U", "T")
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in normalized)


def load_frozen_ribospan(name: str, weight_dir: Path, device: torch.device):
    Config, Model, Tokenizer, vocab = _import_ribospan()
    tokenizer = Tokenizer(str(vocab))
    config = Config.from_pretrained(str(weight_dir), local_files_only=True)
    model = Model.from_pretrained(str(weight_dir), config=config, local_files_only=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    return model, tokenizer


def embed_cls_nucleotides(loaded, sequence_u: str) -> np.ndarray:
    """Return float16 ``(1 + L, H)`` for the structRFM head's leading dummy base."""

    model, tokenizer = loaded
    normalized = normalize_sequence(sequence_u)
    token_ids = [tokenizer.token_to_id(base) for base in normalized]
    unknown = tokenizer.unk_token_id
    token_ids = [unknown if token is None else token for token in token_ids]
    input_ids = torch.tensor(
        [[tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id]],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    attention = torch.ones_like(input_ids)
    encoder = model.ribospan
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
            use_cache=False,
        )
    # Drop SEP. Position 0 is CLS, aligned with SparseEmbedding's leading "0".
    states = outputs.hidden_states[-1][0, :-1]
    array = states.detach().to(device="cpu", dtype=torch.float16).numpy()
    if array.shape[0] != len(normalized) + 1:
        raise RuntimeError(
            f"expected {len(normalized) + 1} positions, got {array.shape[0]}"
        )
    return array
