# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Flexible +/-1 nucleotide pair scores from the RiNALMo secondary-structure metric."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score

THRESHOLDS = [i / 100 for i in range(1, 30)]


def _relax(pairing: np.ndarray) -> np.ndarray:
    padded = np.pad(pairing, ((1, 1), (1, 1)))
    relaxed = (
        np.roll(padded, 1, -1)
        + np.roll(padded, -1, -1)
        + np.roll(padded, 1, -2)
        + np.roll(padded, -1, -2)
    )
    return np.clip(padded + relaxed, 0, 1)[..., 1:-1, 1:-1]


def ss_precision(target: np.ndarray, prediction: np.ndarray, *, flexible: bool = True) -> float:
    if flexible:
        target = _relax(target)
    rows, cols = np.triu_indices(target.shape[-1], k=1)
    return float(
        precision_score(target[rows, cols], prediction[rows, cols], zero_division=0.0)
    )


def ss_recall(target: np.ndarray, prediction: np.ndarray, *, flexible: bool = True) -> float:
    if flexible:
        prediction = _relax(prediction)
    rows, cols = np.triu_indices(target.shape[-1], k=1)
    return float(
        recall_score(target[rows, cols], prediction[rows, cols], zero_division=0.0)
    )


def ss_f1(target: np.ndarray, prediction: np.ndarray, *, flexible: bool = True) -> float:
    precision = ss_precision(target, prediction, flexible=flexible)
    recall = ss_recall(target, prediction, flexible=flexible)
    if precision + recall < 1e-5:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def assert_protocol_cases() -> None:
    """A one-nucleotide shift scores as a hit; the same matrices miss under exact matching."""

    length = 6
    target = np.zeros((length, length), dtype=np.float32)
    prediction = np.zeros((length, length), dtype=np.float32)
    target[0, 4] = target[4, 0] = 1.0
    prediction[0, 5] = prediction[5, 0] = 1.0
    if ss_f1(target, prediction, flexible=True) != 1.0:
        raise AssertionError("a +/-1 shift must be a flexible true positive")
    if ss_f1(target, prediction, flexible=False) != 0.0:
        raise AssertionError("a +/-1 shift must miss under exact matching")
    if ss_f1(target, target, flexible=False) != 1.0:
        raise AssertionError("an exact copy must score 1")
