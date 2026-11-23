#!/usr/bin/env python3
"""Model-size scaling figure: zero-shot SS quality (macro-avg NormAUPR across
7 eval sets, epoch6 baseline) vs LIMBER model size. Bold/minimal for a
main-figure panel — no fine print, thick line, large markers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _paths import REPO

OUT = str(REPO / "outputs")
df = pd.read_csv(f"{OUT}/fig4_modelsize_normaupr.csv")

x = df["n_params"].to_numpy()
y = df["macro_avg_NormAUPR_baseline_ep6"].to_numpy()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(x, y, color="#1E88E5", lw=10, zorder=2, solid_capstyle="round")
ax.scatter(x, y, s=520, color="#1E88E5", edgecolor="white", linewidth=3.5, zorder=3)

ax.set_xscale("log")
ax.set_xticks(x)
ax.set_xticklabels(df["size_label"], fontsize=22, fontweight="bold")
ax.tick_params(axis="x", length=8, width=2.5, pad=14)
ax.tick_params(axis="y", labelsize=18)
ax.set_ylim(0, max(y) * 1.18)
ax.set_xlim(x.min() / 1.6, x.max() * 1.6)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(2.5)
ax.tick_params(width=2.5, length=8)

fig.tight_layout(pad=1.0)
fig.savefig(f"{OUT}/fig4_modelsize_scaling.png", dpi=300, transparent=True)
fig.savefig(f"{OUT}/fig4_modelsize_scaling.pdf", dpi=300)
print("saved:", f"{OUT}/fig4_modelsize_scaling.png/.pdf")
