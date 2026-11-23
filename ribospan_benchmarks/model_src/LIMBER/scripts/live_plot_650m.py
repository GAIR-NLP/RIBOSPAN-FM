#!/usr/bin/env python3
"""Standalone live plotter for the Hybrid-650M run: reads metrics_history.json
(written every 50 steps by the trainer) and redraws loss/LR/PPL curves every
poll interval, without touching the training process."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/fm_hybrid_650m")
SPE = 30278


def plot():
    h = json.load(open(OUT / "metrics_history.json"))
    if not h["step"]:
        return
    ep = [s / SPE for s in h["step"]]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].plot(ep, h["train_loss"], lw=.6, color="tab:blue", label="train")
    if h.get("val_step"):
        ax[0].plot([s / SPE for s in h["val_step"]], h["val_loss"], "o-", color="tab:red", label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MLM loss"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_title(f"MLM loss (step {h['step'][-1]:,}, last {h['train_loss'][-1]:.3f})")
    ax[1].plot(ep, h["lr"], color="tab:green"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("lr")
    ax[1].set_title("learning rate"); ax[1].grid(alpha=.3)
    if h.get("val_ppl"):
        ax[2].plot([s / SPE for s in h["val_step"]], h["val_ppl"], "o-", color="tab:purple")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("per-base PPL"); ax[2].set_title("val per-base PPL"); ax[2].grid(alpha=.3)
    fig.suptitle("Hybrid-650M MLM pretraining — full RNAcentral, ctx 1022, 6 epochs")
    fig.tight_layout(); fig.savefig(OUT / "live_curves.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    while True:
        try:
            plot()
        except Exception as e:
            print(f"[plot] {e}", flush=True)
        time.sleep(120)
