#!/usr/bin/env python3
"""Training-data-fraction comparison: supervised SS fine-tuning macro-F1 vs
pretraining data fraction (20/40/60/80/100%) for LIMBER-100M. Pulled from
existing ss_eval summaries (hybrid_100m_{20,40,60,80}pct + fm_hybrid_100m) —
no fresh eval needed. Same bold/minimal style, third accent color."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _paths import REPO

OUT = str(REPO / "outputs")
df = pd.read_csv(f"{OUT}/fig4_datafrac_scaling_f1.csv")

x = df["data_pct"].to_numpy()
y = df["macro_f1"].to_numpy()

COLOR = "#00897B"  # vivid teal (distinct from the blue model-size and purple epoch plots)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(x, y, color=COLOR, lw=10, zorder=2, solid_capstyle="round")
ax.scatter(x, y, s=520, color=COLOR, edgecolor="white", linewidth=3.5, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([f"{int(v)}%" for v in x], fontsize=22, fontweight="bold")
ax.tick_params(axis="x", length=8, width=2.5, pad=14)
ax.tick_params(axis="y", labelsize=18)
ax.set_ylim(0, max(y) * 1.18)
ax.set_xlim(x.min() - 8, x.max() + 8)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(2.5)
ax.tick_params(width=2.5, length=8)

fig.tight_layout(pad=1.0)
fig.savefig(f"{OUT}/fig4_datafrac_scaling.png", dpi=300, transparent=True)
fig.savefig(f"{OUT}/fig4_datafrac_scaling.pdf", dpi=300)
print("saved:", f"{OUT}/fig4_datafrac_scaling.png/.pdf")
