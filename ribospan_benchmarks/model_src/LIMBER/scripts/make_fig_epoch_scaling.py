#!/usr/bin/env python3
"""Training-epoch comparison: RNAcentral validation perplexity vs training
epoch (3/6/9/12) for LIMBER-650M. Pulled directly from the pretraining run's
periodic [val] log lines (no fresh eval needed — categorical-Jacobian NormAUPR
baseline reruns were vetoed as too slow). Same bold/minimal style as the
model-size scaling figure, different accent color."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _paths import REPO

OUT = str(REPO / "outputs")
df = pd.read_csv(f"{OUT}/fig4_epoch_scaling_ppl.csv")

x = df["epoch"].to_numpy()
y = df["val_ppl_rnacentral"].to_numpy()

COLOR = "#8E24AA"  # vivid purple (different from the model-size plot's blue)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(x, y, color=COLOR, lw=10, zorder=2, solid_capstyle="round")
ax.scatter(x, y, s=520, color=COLOR, edgecolor="white", linewidth=3.5, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([str(e) for e in x], fontsize=22, fontweight="bold")
ax.tick_params(axis="x", length=8, width=2.5, pad=14)
ax.tick_params(axis="y", labelsize=18)
pad = (y.max() - y.min()) * 0.35
ax.set_ylim(y.min() - pad, y.max() + pad)
ax.set_xlim(x.min() - 1.5, x.max() + 1.5)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(2.5)
ax.tick_params(width=2.5, length=8)

fig.tight_layout(pad=1.0)
fig.savefig(f"{OUT}/fig4_epoch_scaling.png", dpi=300, transparent=True)
fig.savefig(f"{OUT}/fig4_epoch_scaling.pdf", dpi=300)
print("saved:", f"{OUT}/fig4_epoch_scaling.png/.pdf")
