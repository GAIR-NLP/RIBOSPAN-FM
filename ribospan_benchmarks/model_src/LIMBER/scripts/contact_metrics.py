"""Contact-map evaluation metrics for unsupervised RNA contact probing.

Given a symmetric score matrix S [L,L], a gold base-pair list, and the
sequence, compute AUPRC / AUROC / Precision@{L, L/2, num_gold} / best-F1 /
best-precision / best-recall, plus canonical-pair-restricted AUPRC & F1.

Evaluation rules: only i<j candidate pairs, exclude |i-j| < min_sep (default 4).
Negatives = all candidate pairs that are not gold. Sequences with 0 gold pairs
are skipped (caller records a warning).

Canonical pairs: A-U, U-A, G-C, C-G, G-U, U-G.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
             ("G", "U"), ("U", "G")}


def _candidate_mask(L, min_sep):
    ii, jj = np.triu_indices(L, k=min_sep)   # i<j and j-i>=min_sep
    return ii, jj


def _best_f1(labels, scores):
    """Sweep thresholds (at the score values) for best F1; return f1,prec,rec."""
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = y.sum()
    if n_pos == 0:
        return 0.0, 0.0, 0.0
    prec = tp / (tp + fp)
    rec = tp / n_pos
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    k = int(np.argmax(f1))
    return float(f1[k]), float(prec[k]), float(rec[k])


def _precision_at(labels, scores, k):
    if k <= 0:
        return float("nan")
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    return float(labels[top].sum() / k)


def supervised_metrics(probs, pairs, sequence, min_sep=4, threshold=0.5):
    """For a predicted probability matrix: threshold P/R/F1/MCC (at `threshold`)
    plus threshold-free AUPRC/AUROC/P@{L,L/2,num_gold}/best_F1. Only i<j with
    j-i>=min_sep are scored. Returns {'skipped':True} if no gold pairs."""
    L = len(sequence)
    gold = {(min(i, j), max(i, j)) for i, j in pairs if abs(i - j) >= min_sep}
    if not gold:
        return {"skipped": True, "reason": "no gold pairs"}
    ii, jj = _candidate_mask(L, min_sep)
    scores = probs[ii, jj].astype(np.float64)
    labels = np.array([(int(i), int(j)) in gold for i, j in zip(ii, jj)], dtype=np.int32)
    n_gold = int(labels.sum())

    pred = (scores >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    bf1, bp, br = _best_f1(labels, scores)
    out = {
        "skipped": False, "length": L, "num_gold_pairs": n_gold,
        "precision": prec, "recall": rec, "F1": f1, "MCC": mcc,
        "AUPRC": float(average_precision_score(labels, scores)),
        "AUROC": float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else float("nan"),
        "P_at_L": _precision_at(labels, scores, L),
        "P_at_L2": _precision_at(labels, scores, L // 2),
        "P_at_num_gold": _precision_at(labels, scores, n_gold),
        "best_F1": bf1, "best_precision": bp, "best_recall": br,
    }
    return out


def evaluate(S, pairs, sequence, min_sep=4):
    """Return a metrics dict, or {'skipped':True,...} if no gold pairs."""
    L = len(sequence)
    gold = set()
    for i, j in pairs:
        a, b = (i, j) if i < j else (j, i)
        if b - a >= min_sep:
            gold.add((a, b))
    if not gold:
        return {"skipped": True, "reason": "no gold pairs (after min_sep filter)"}

    ii, jj = _candidate_mask(L, min_sep)
    scores = S[ii, jj].astype(np.float64)
    labels = np.array([(int(i), int(j)) in gold for i, j in zip(ii, jj)], dtype=np.int32)

    # canonical-pair restriction
    canon = np.array([(sequence[i], sequence[j]) in CANONICAL
                      for i, j in zip(ii, jj)], dtype=bool)

    n_gold = int(labels.sum())
    out = {
        "skipped": False, "length": L, "num_gold_pairs": n_gold,
        "num_candidates": int(len(labels)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "AUROC": float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else float("nan"),
        "P_at_L": _precision_at(labels, scores, L),
        "P_at_L2": _precision_at(labels, scores, L // 2),
        "P_at_num_gold": _precision_at(labels, scores, n_gold),
    }
    f1, prec, rec = _best_f1(labels, scores)
    out["best_F1"], out["best_precision"], out["best_recall"] = f1, prec, rec

    # canonical-only (negatives = non-gold canonical candidates)
    if canon.sum() > 0 and labels[canon].sum() > 0:
        cl, cs = labels[canon], scores[canon]
        out["canonical_pair_AUPRC"] = float(average_precision_score(cl, cs)) \
            if 0 < cl.sum() < len(cl) else float("nan")
        cf1, _, _ = _best_f1(cl, cs)
        out["canonical_pair_F1"] = cf1
    else:
        out["canonical_pair_AUPRC"] = float("nan")
        out["canonical_pair_F1"] = float("nan")
    return out
