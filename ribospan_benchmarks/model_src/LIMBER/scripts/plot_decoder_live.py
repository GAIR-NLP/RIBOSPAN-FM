#!/usr/bin/env python3
"""Overlay the two decoder-only (causal LM) runs — kmer1 and kmer6 — into one
figure: train loss, eval loss, and eval next-token accuracy. Reads each run's
metrics_history.json (written by the shared LiveChartCallback). Safe to call
repeatedly while training is in progress.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "outputs"
SAVE = OUT / "fm_decoder_live.png"
RUNS = [
    ("fm_decoder_kmer1", "decoder kmer1", "tab:blue"),
    ("fm_decoder_kmer6", "decoder kmer6", "tab:orange"),
]


def load(run):
    p = OUT / run / "metrics_history.json"
    return json.load(open(p)) if p.exists() else None


def main():
    fig, (ax_l, ax_e, ax_a) = plt.subplots(1, 3, figsize=(17, 5))
    for run, label, color in RUNS:
        h = load(run)
        if not h:
            continue
        if h["train"]["step"]:
            ax_l.plot(h["train"]["step"], h["train"]["loss"], color=color, alpha=0.8, label=label)
        if h["eval"]["step"]:
            ax_e.plot(h["eval"]["step"], h["eval"]["loss"], color=color, marker="o", ms=3, label=label)
            acc = [a * 100 for a in h["eval"]["masked_accuracy"]]
            ax_a.plot(h["eval"]["step"], acc, color=color, marker="o", ms=3, label=label)
            if acc:
                ax_a.annotate(f"{acc[-1]:.1f}%", (h["eval"]["step"][-1], acc[-1]),
                              textcoords="offset points", xytext=(4, 0), color=color, fontsize=9)
    for ax, t, yl in ((ax_l, "train loss (causal LM)", "loss"),
                      (ax_e, "eval loss (causal LM)", "eval loss"),
                      (ax_a, "eval next-token accuracy", "accuracy (%)")):
        ax.set_xlabel("step"); ax.set_ylabel(yl); ax.set_title(t)
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("RNA decoder-only (next-token) pretraining — kmer1 vs kmer6 (live)", fontsize=13)
    fig.tight_layout()
    fig.savefig(SAVE, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {SAVE}")


if __name__ == "__main__":
    main()
