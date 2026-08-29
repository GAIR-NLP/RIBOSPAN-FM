# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Concat nucleotide states, then mean-pool."""

from __future__ import annotations

import numpy as np


def last_layer_index(loaded) -> int:
    """Match RNA-type extract.py: final transformer block, not the embedding."""

    return int(loaded.num_hidden_layers)


def chunk_sequence(sequence: str, chunk_length: int) -> list[str]:
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    return [sequence[i : i + chunk_length] for i in range(0, len(sequence), chunk_length)]


def weighted_mean(vectors: list[np.ndarray], weights: list[int]) -> np.ndarray:
    """Length-weighted mean; equal to mean of concatenated nucleotide rows."""

    if not vectors:
        raise ValueError("no chunk vectors")
    matrix = np.stack([np.asarray(vec, dtype=np.float64) for vec in vectors], axis=0)
    scale = np.asarray(weights, dtype=np.float64)
    if scale.shape[0] != matrix.shape[0]:
        raise ValueError("weights and vectors have different lengths")
    total = float(scale.sum())
    if total <= 0:
        raise ValueError("chunk weights must be positive")
    pooled = (matrix * scale[:, None]).sum(axis=0) / total
    return pooled.astype(np.float32)


def embed_sequence_mean(loaded, sequence: str) -> tuple[np.ndarray, int]:
    """Return (H,) mean-pooled last-layer embedding and the number of chunks.

    Sequences longer than ``loaded.effective_max_length`` are split into
    non-overlapping native-window chunks. Per-chunk ``mean`` pooling plus
    length weighting is algebraically identical to concatenating nucleotide
    hidden states and then averaging, which is the mRNABench protocol.
    """

    from scripts.backends import normalize_sequence

    seq = normalize_sequence(sequence)
    chunk_len = int(loaded.effective_max_length)
    layer = last_layer_index(loaded)
    chunks = chunk_sequence(seq, chunk_len) if len(seq) > chunk_len else [seq]
    vectors: list[np.ndarray] = []
    weights: list[int] = []
    for chunk in chunks:
        pooled = loaded.embed_pooled(chunk, layers=[layer], poolings=["mean"])
        vectors.append(np.asarray(pooled[layer]["mean"], dtype=np.float32))
        weights.append(len(chunk))
    if len(vectors) == 1:
        return vectors[0], 1
    return weighted_mean(vectors, weights), len(vectors)
