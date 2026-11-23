#!/usr/bin/env python3
"""Aggregate per-(model,method) contact-eval summaries into one comparison
table + barplot. Reads metrics_summary.json from each
outputs/contact_eval/{model}/{method}/ dir listed on the command line.

Writes outputs/contact_eval/compare_summary.{csv,json} and compare_barplot.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "contact_eval"

COLUMNS = ["model", "method", "dataset", "num_sequences",
           "mean_AUPRC", "mean_AUROC", "mean_P_at_L", "mean_P_at_L2",
           "mean_P_at_num_gold", "mean_best_F1", "median_AUPRC", "median_best_F1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entries", nargs="+", required=True,
                   help="model=path/to/method_dir triples, e.g. kmer1-BERT=outputs/contact_eval/kmer1-BERT/categorical_jacobian")
    p.add_argument("--dataset", default="archiveII")
    args = p.parse_args()

    rows = []
    for ent in args.entries:
        model, d = ent.split("=", 1)
        sm = json.load(open(Path(d) / "metrics_summary.json"))
        rows.append({
            "model": model, "method": sm.get("method", Path(d).name),
            "dataset": args.dataset, "num_sequences": sm.get("num_sequences", 0),
            "mean_AUPRC": sm.get("mean_AUPRC"), "mean_AUROC": sm.get("mean_AUROC"),
            "mean_P_at_L": sm.get("mean_P_at_L"), "mean_P_at_L2": sm.get("mean_P_at_L2"),
            "mean_P_at_num_gold": sm.get("mean_P_at_num_gold"),
            "mean_best_F1": sm.get("mean_best_F1"),
            "median_AUPRC": sm.get("median_AUPRC"),
            "median_best_F1": sm.get("median_best_F1"),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "compare_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(OUT / "compare_summary.json", "w"), indent=2)

    # grouped barplot: AUPRC, P@L, best_F1 per (model,method)
    labels = [f"{r['model']}\n{r['method']}" for r in rows]
    metrics = [("mean_AUPRC", "mean AUPRC"), ("mean_P_at_L", "mean P@L"),
               ("mean_best_F1", "mean best-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] or 0 for r in rows]
        bars = ax.bar(range(len(rows)), vals,
                      color=["tab:blue", "tab:cyan", "tab:orange", "tab:red"][:len(rows)])
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Unsupervised RNA contact probing — model × method (ArchiveII)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "compare_barplot.png", dpi=120)
    print(f"[aggregate] wrote {OUT}/compare_summary.{{csv,json}} and compare_barplot.png")
    for r in rows:
        print(f"  {r['model']:14}{r['method']:22} AUPRC={r['mean_AUPRC']:.3f} "
              f"P@L={r['mean_P_at_L']:.3f} F1={r['mean_best_F1']:.3f}")


if __name__ == "__main__":
    main()
