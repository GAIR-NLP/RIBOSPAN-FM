#!/usr/bin/env python3
"""Overlay the eval ACC / loss curves of multiple tokenizer runs for comparison.

Reads each run's eval_metrics.csv and renders a single side-by-side chart:
masked-token accuracy (left) and eval loss (right). Highlights the overlapping
center3 leakage spike vs the clean single / non-overlapping kmer3 curves.

Run:
    python scripts/plot_compare.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "bert_mlm"

# Auto-discover the 100k runs present on disk (single + kmer sweep), in a
# sensible k order, and assign distinct colors.
_CMAP = plt.get_cmap("tab10")


def discover_runs():
    runs = []
    order = ["single_100k"] + [f"kmer{k}_100k" for k in (2, 3, 4, 5, 6)]
    i = 0
    for name in order:
        if (OUT / name / "eval_metrics.csv").exists():
            label = name.replace("_100k", "")
            runs.append((name, label, _CMAP(i % 10)))
            i += 1
    return runs


RUNS = discover_runs()


def load(run: str):
    path = OUT / run / "eval_metrics.csv"
    if not path.exists():
        return None
    steps, loss, acc = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"]))
            loss.append(float(row["eval_loss"]))
            acc.append(float(row["masked_accuracy"]) * 100)
    return steps, loss, acc


def main():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))
    for run, label, color in RUNS:
        data = load(run)
        if data is None:
            continue
        steps, loss, acc = data
        ax_acc.plot(steps, acc, color=color, marker="o", ms=3, label=label)
        ax_loss.plot(steps, loss, color=color, marker="o", ms=3, label=label)
        ax_acc.annotate(f"{acc[-1]:.1f}%", xy=(steps[-1], acc[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        color=color, fontsize=9)

    ax_acc.set_title("Masked-token prediction success rate")
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("masked-token accuracy (%) [token-level, NOT cross-k comparable]")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="center right", fontsize=9)

    ax_loss.set_title("Eval MLM loss")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("eval loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=9)

    fig.suptitle("RNA BERT MLM — tokenizer comparison", fontsize=13)
    fig.tight_layout()
    out_path = OUT / "compare_acc.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
