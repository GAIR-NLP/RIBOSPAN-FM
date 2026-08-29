# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""RNAGym fitness metrics: abs Spearman, folded AUC, abs MCC."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def _finite_pairs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(true) & np.isfinite(score)
    return true[mask], score[mask]


def spearman_abs(y_true: np.ndarray, y_score: np.ndarray) -> float:
    true, score = _finite_pairs(y_true, y_score)
    if true.size < 3:
        return float("nan")
    if np.std(true) == 0 or np.std(score) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        corr = spearmanr(true, score).correlation
    except Exception:
        ranks_t = true.argsort().argsort().astype(np.float64)
        ranks_s = score.argsort().argsort().astype(np.float64)
        corr = float(np.corrcoef(ranks_t, ranks_s)[0, 1])
    if corr is None or not math.isfinite(float(corr)):
        return float("nan")
    return abs(float(corr))


def auc_folded(y_true: np.ndarray, y_score: np.ndarray) -> float:
    true, score = _finite_pairs(y_true, y_score)
    if true.size < 3:
        return float("nan")
    labels = (true > np.median(true)).astype(np.int64)
    if labels.min() == labels.max():
        return float("nan")
    try:
        auc = float(roc_auc_score(labels, score))
    except ValueError:
        return float("nan")
    return max(auc, 1.0 - auc)


def mcc_abs(y_true: np.ndarray, y_score: np.ndarray) -> float:
    true, score = _finite_pairs(y_true, y_score)
    if true.size < 3:
        return float("nan")
    labels = (true > np.median(true)).astype(np.int64)
    preds = (score > np.median(score)).astype(np.int64)
    if labels.min() == labels.max() or preds.min() == preds.max():
        return float("nan")
    return abs(float(matthews_corrcoef(labels, preds)))


def assay_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "Spearman": spearman_abs(y_true, y_score),
        "AUC": auc_folded(y_true, y_score),
        "MCC": mcc_abs(y_true, y_score),
        "n": int(np.isfinite(np.asarray(y_true, dtype=np.float64)).sum()),
        "n_scored": int(
            (
                np.isfinite(np.asarray(y_true, dtype=np.float64))
                & np.isfinite(np.asarray(y_score, dtype=np.float64))
            ).sum()
        ),
    }
