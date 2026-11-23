#!/usr/bin/env python3
"""Categorical-Jacobian contact extraction from a (frozen) RNA FM — kmer1.

For each sequence x of length L over alphabet {A,U,C,G,N}:
  1. forward wild-type -> logits_wt[L, A]
  2. for every position i and alt base a: forward x^{i->a} -> logits[L, A]
     J[i,a,j,b] = logits^{i->a}[j,b] - logits_wt[j,b]
  3. pair score S[i,j] = ||J[i,:,j,:]||_F   (Frobenius over a,b)
  4. symmetrize, zero diagonal, APC correction
  5. metrics (|i-j|>=4, i<j) + save scores/{id}.npy (post-APC) + plots

O(L * |alphabet|) forwards per sequence (batched). Resumable: skips sequences
whose scores npy already exists. First version is kmer1 only.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc          # noqa: E402
from contact_metrics import evaluate  # noqa: E402


@torch.no_grad()
def jacobian_scores(model, tokenizer, seq, device, batch_size):
    L = len(seq)
    base_ids = cc.base_token_ids(tokenizer)            # 5 ids
    wt_ids, am = cc.encode_kmer1(tokenizer, seq, device)   # [1, L+2]
    wt_row = wt_ids[0].clone()

    logits_wt = model(input_ids=wt_ids, attention_mask=am).logits[0]  # [L+2, V]
    base_cols = torch.tensor(base_ids, device=device)
    wt_base = logits_wt[1:1 + L][:, base_cols].float()      # [L, 5]

    # build all 5L mutated rows; remember each row's source position i
    rows, src_i = [], []
    for i in range(L):
        for a_id in base_ids:
            r = wt_row.clone()
            r[i + 1] = a_id
            rows.append(r)
            src_i.append(i)
    rows = torch.stack(rows)                                # [5L, L+2]
    src_i = np.asarray(src_i)

    S2 = np.zeros((L, L), dtype=np.float64)
    for s in range(0, rows.shape[0], batch_size):
        chunk = rows[s:s + batch_size]
        amb = torch.ones_like(chunk)
        logits = model(input_ids=chunk, attention_mask=amb).logits  # [b, L+2, V]
        d = logits[:, 1:1 + L][:, :, base_cols].float() - wt_base    # [b, L, 5]
        contrib = (d ** 2).sum(dim=-1).cpu().numpy()                 # [b, L] = sum_b delta^2
        for r in range(chunk.shape[0]):
            S2[src_i[s + r]] += contrib[r]
    return np.sqrt(S2)                                      # S[i,j] = ||J[i,:,j,:]||_F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_type", choices=["bert", "hybrid"], required=True)
    p.add_argument("--tokenizer_type", default="kmer1")
    p.add_argument("--vocab_dir", default="tokenizers/single")
    p.add_argument("--dataset_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max-seqs", type=int, default=100, dest="max_seqs")
    p.add_argument("--max-len", type=int, default=256, dest="max_len")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--num-plots", type=int, default=10, dest="num_plots")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.tokenizer_type != "kmer1":
        raise SystemExit("first version supports --tokenizer_type kmer1 only")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = cc.load_model(args.model_dir, args.model_type, device)
    tokenizer = cc.load_tokenizer(args.vocab_dir)

    out = Path(args.output_dir)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "example_plots").mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(args.dataset_jsonl)]
    recs = [r for r in recs if r["length"] <= args.max_len][:args.max_seqs]

    rows_csv, n_plot, skipped = [], 0, 0
    for k, rec in enumerate(recs):
        npy = out / "scores" / f"{rec['id']}.npy"
        if npy.exists():
            S_apc = np.load(npy)
            S_sym = None
        else:
            S_raw = jacobian_scores(model, tokenizer, rec["sequence"], device, args.batch_size)
            S_sym, S_apc = cc.finalize(S_raw)
            np.save(npy, S_apc)
        m = evaluate(S_apc, rec["pairs"], rec["sequence"])
        if m.get("skipped"):
            skipped += 1
            continue
        m_row = {"id": rec["id"], "family": rec["family"], "length": rec["length"], **m}
        rows_csv.append(m_row)
        if n_plot < args.num_plots and S_sym is not None:
            cc.plot_example(out / "example_plots" / f"{rec['id']}.png", rec,
                            S_sym, S_apc, auprc=m["AUPRC"], f1=m["best_F1"])
            n_plot += 1
        print(f"[{k+1}/{len(recs)}] {rec['id']:42} AUPRC={m['AUPRC']:.3f} "
              f"P@L={m['P_at_L']:.3f} F1={m['best_F1']:.3f}")

    # per-sequence csv
    if rows_csv:
        with open(out / "metrics_per_sequence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)

    def agg(key, fn):
        vals = [r[key] for r in rows_csv if r.get(key) == r.get(key)]  # drop NaN
        return fn(vals) if vals else float("nan")

    summary = {"method": "categorical_jacobian", "model_dir": args.model_dir,
               "model_type": args.model_type, "dataset": args.dataset_jsonl,
               "num_sequences": len(rows_csv), "num_skipped": skipped}
    for key in ("AUPRC", "AUROC", "P_at_L", "P_at_L2", "P_at_num_gold",
                "best_F1", "best_precision", "best_recall",
                "canonical_pair_AUPRC", "canonical_pair_F1"):
        summary[f"mean_{key}"] = agg(key, statistics.mean)
        summary[f"median_{key}"] = agg(key, statistics.median)
    json.dump(summary, open(out / "metrics_summary.json", "w"), indent=2)

    print(f"\n[done] {len(rows_csv)} seqs ({skipped} skipped) -> {out}")
    print(f"  mean AUPRC={summary['mean_AUPRC']:.4f}  P@L={summary['mean_P_at_L']:.4f}  "
          f"best_F1={summary['mean_best_F1']:.4f}")


if __name__ == "__main__":
    main()
