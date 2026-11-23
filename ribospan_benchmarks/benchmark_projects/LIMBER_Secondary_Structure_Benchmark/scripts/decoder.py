# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Canonical-pair greedy decoder from the RiNALMo secondary-structure pipeline."""

from __future__ import annotations

import numpy as np

CANONICAL_PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}
SHARP_LOOP_DIST = 4
_BASE_INDEX = {"A": 0, "U": 1, "C": 2, "G": 3, "N": 4}
_PAIR_OK = np.zeros((5, 5), dtype=bool)
for _left, _right in CANONICAL_PAIRS:
    _PAIR_OK[_BASE_INDEX[_left], _BASE_INDEX[_right]] = True


def _sharp_loop_mask(length: int) -> np.ndarray:
    mask = np.eye(length, dtype=bool)
    for offset in range(1, SHARP_LOOP_DIST):
        mask |= np.eye(length, k=offset, dtype=bool) | np.eye(length, k=-offset, dtype=bool)
    return mask


def _canonical_mask(sequence: str) -> np.ndarray:
    index = np.fromiter(
        (_BASE_INDEX.get(base, 4) for base in sequence),
        dtype=np.intp,
        count=len(sequence),
    )
    return _PAIR_OK[index[:, None], index[None, :]]


def _greedy_clean(pairing: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clean = np.copy(pairing)
    scores = np.copy(probabilities)
    scores[pairing < 1] = 0.0
    while np.sum(scores > 0.0) > 0:
        row, col = np.unravel_index(np.argmax(scores), scores.shape)
        scores[row, :] = scores[col, :] = scores[:, row] = scores[:, col] = 0.0
        clean[row, :] = clean[col, :] = clean[:, row] = clean[:, col] = 0
        clean[row, col] = clean[col, row] = 1
    return clean


def prob_mat_to_sec_struct(
    probabilities: np.ndarray,
    sequence: str,
    threshold: float = 0.5,
    *,
    allow_noncanonical: bool = False,
    allow_sharp: bool = False,
) -> np.ndarray:
    scores = np.array(probabilities, dtype=np.float64)
    length = scores.shape[-1]
    allowed = ~np.eye(length, dtype=bool)
    if not allow_sharp:
        allowed &= ~_sharp_loop_mask(length)
    if not allow_noncanonical:
        allowed &= _canonical_mask(sequence)
    scores = scores.copy()
    scores[~allowed] = 0.0
    pairing = (scores > threshold).astype(int)
    return _greedy_clean(pairing, scores)
