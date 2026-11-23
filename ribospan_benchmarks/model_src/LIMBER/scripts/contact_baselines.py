#!/usr/bin/env python3
"""Baselines + recheck for the contact-probing audit (Tasks 7/8).

Evaluates, on the same ArchiveII subset and metric, three reference baselines:
  - random           : symmetric random matrix
  - distance         : S[i,j] = -|i-j|  (pure sequence-distance prior)
  - canonical_random : random score on canonical/wobble candidates only, low elsewhere
plus any trained-model score dirs passed as name=scores_dir (loads cached
{id}.npy = post-APC model scores). Writes a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_metrics import evaluate, CANONICAL


def random_S(L, rng):
    S = rng.rand(L, L); S = (S + S.T) / 2; np.fill_diagonal(S, 0); return S


def distance_S(L):
    idx = np.arange(L)
    return -np.abs(idx[:, None] - idx[None, :]).astype(float)


def canonical_random_S(seq, rng):
    L = len(seq)
    S = np.full((L, L), -10.0)
    for i in range(L):
        for j in range(i + 1, L):
            if (seq[i], seq[j]) in CANONICAL:
                S[i, j] = S[j, i] = rng.rand()
    np.fill_diagonal(S, 0)
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_jsonl", default="data/contact_eval/archiveII.jsonl")
    p.add_argument("--max-seqs", type=int, default=100, dest="max_seqs")
    p.add_argument("--max-len", type=int, default=256, dest="max_len")
    p.add_argument("--model-scores", nargs="*", default=[],
                   help="name=path/to/scores_dir (loads {id}.npy = post-APC model scores)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    recs = [json.loads(l) for l in open(args.dataset_jsonl)]
    recs = [r for r in recs if r["length"] <= args.max_len][:args.max_seqs]

    model_dirs = {}
    for ms in args.model_scores:
        name, d = ms.split("=", 1)
        model_dirs[name] = Path(d)

    methods = ["random", "distance", "canonical_random"] + list(model_dirs)
    acc = {m: {"AUPRC": [], "P_at_L": [], "best_F1": []} for m in methods}

    for r in recs:
        seq, L = r["sequence"], r["length"]
        cand = {
            "random": random_S(L, rng),
            "distance": distance_S(L),
            "canonical_random": canonical_random_S(seq, rng),
        }
        for name, d in model_dirs.items():
            npy = d / f"{r['id']}.npy"
            cand[name] = np.load(npy) if npy.exists() else None
        for m in methods:
            S = cand.get(m)
            if S is None:
                continue
            res = evaluate(S, r["pairs"], seq)
            if res.get("skipped"):
                continue
            for k in ("AUPRC", "P_at_L", "best_F1"):
                acc[m][k].append(res[k])

    rows = []
    for m in methods:
        a = acc[m]
        n = len(a["AUPRC"])
        rows.append({
            "method": m, "num_sequences": n,
            "mean_AUPRC": round(statistics.mean(a["AUPRC"]), 4) if n else float("nan"),
            "mean_P_at_L": round(statistics.mean(a["P_at_L"]), 4) if n else float("nan"),
            "mean_best_F1": round(statistics.mean(a["best_F1"]), 4) if n else float("nan"),
            "median_AUPRC": round(statistics.median(a["AUPRC"]), 4) if n else float("nan"),
            "median_best_F1": round(statistics.median(a["best_F1"]), 4) if n else float("nan"),
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[baselines] wrote {args.out_csv}  ({len(recs)} seqs)")
    for r in rows:
        print(f"  {r['method']:24} AUPRC={r['mean_AUPRC']:.4f} P@L={r['mean_P_at_L']:.4f} "
              f"F1={r['mean_best_F1']:.4f}  medAUPRC={r['median_AUPRC']:.4f}")


if __name__ == "__main__":
    main()
