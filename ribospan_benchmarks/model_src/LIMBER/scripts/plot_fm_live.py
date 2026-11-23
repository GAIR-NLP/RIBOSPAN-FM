#!/usr/bin/env python3
"""Overlay the live training/eval curves of the two foundation-model runs
(kmer1 / single and kmer6) into one figure. Reads each run's
metrics_history.json (written by the training LiveChartCallback) and draws:
train loss, eval loss, and eval masked-token accuracy. Safe to call repeatedly
while training is in progress (e.g. from a watch loop).

Run:
    python scripts/plot_fm_live.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTROOT = Path(__file__).resolve().parent.parent / "outputs"
SAVE_PATH = OUTROOT / "bert_mlm" / "fm_live.png"
# (relative-to-outputs path, label, color)
RUNS = [
    ("bert_mlm/fm_single", "kmer1 BERT", "tab:blue"),
    ("fm_hybrid_mamba_kmer1", "kmer1 hybrid (T+Mamba)", "tab:cyan"),
    ("bert_mlm/fm_kmer6",  "kmer6 BERT", "tab:green"),
    ("fm_hybrid_mamba_kmer6", "kmer6 hybrid (T+Mamba)", "tab:orange"),
]


def load(run):
    p = OUTROOT / run / "metrics_history.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    fig, (ax_l, ax_e, ax_a) = plt.subplots(1, 3, figsize=(17, 5))
    any_data = False
    for run, label, color in RUNS:
        h = load(run)
        if not h:
            continue
        any_data = True
        ts, tl = h["train"]["step"], h["train"]["loss"]
        es = h["eval"]["step"]
        el = h["eval"]["loss"]
        ea = [a * 100 for a in h["eval"]["masked_accuracy"]]
        if ts:
            ax_l.plot(ts, tl, color=color, alpha=0.8, label=label)
        if es:
            ax_e.plot(es, el, color=color, marker="o", ms=3, label=label)
            ax_a.plot(es, ea, color=color, marker="o", ms=3, label=label)
            if ea:
                ax_a.annotate(f"{ea[-1]:.1f}%", (es[-1], ea[-1]),
                              textcoords="offset points", xytext=(4, 0),
                              color=color, fontsize=9)

    for ax, t, yl in ((ax_l, "train loss", "MLM loss"),
                      (ax_e, "eval loss", "eval MLM loss"),
                      (ax_a, "eval masked-token accuracy", "accuracy (%)")):
        ax.set_xlabel("step")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("RNA FM pretraining (live) — kmer1 BERT vs kmer6 BERT vs kmer1 hybrid (T+Mamba)",
                 fontsize=12)
    fig.tight_layout()
    out = SAVE_PATH
    fig.savefig(out, dpi=110)
    plt.close(fig)
    if any_data:
        print(f"[plot] wrote {out}")
    else:
        print("[plot] no metrics yet")


if __name__ == "__main__":
    main()
