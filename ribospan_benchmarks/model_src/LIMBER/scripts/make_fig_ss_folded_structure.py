#!/usr/bin/env python3
"""Folded secondary-structure figure for the example tRNA (bpRNA_SPR_10):
real 2D fold layout via ViennaRNA's get_xy_coordinates, TP/FP/FN color-coded,
thick bold strokes for main-figure legibility at small size."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import RNA
from _paths import REPO

OUT = str(REPO / "outputs")
rec = json.load(open(f"{OUT}/example_trna_pairs.json"))
dbn = rec["dbn"]
true_pairs = [tuple(p) for p in rec["true_pairs"]]
tp = [tuple(p) for p in rec["tp"]]
fp = [tuple(p) for p in rec["fp"]]
L = len(dbn)

coords = RNA.get_xy_coordinates(dbn)
xs = np.array([coords.get(i).X for i in range(L)])
ys = -np.array([coords.get(i).Y for i in range(L)])

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(xs, ys, color="#222222", lw=7.0, zorder=1, solid_capstyle="round")
for (i, j) in true_pairs:
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#BFBFBF", lw=5.0, zorder=2)
for (i, j) in tp:
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#00C853", lw=8.5, zorder=4, solid_capstyle="round")
for (i, j) in fp:
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#FF1744", lw=7.0, zorder=5,
              linestyle=(0, (2.2, 1.8)), solid_capstyle="round")
ax.scatter(xs, ys, s=260, color="#1A1A1A", zorder=3, edgecolor="white", linewidth=2.2)
ax.set_aspect("equal")
ax.axis("off")

fig.savefig(f"{OUT}/fig_ss_folded_structure.png", dpi=300, transparent=True, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_ss_folded_structure.pdf", dpi=300, bbox_inches="tight")
print("saved:", f"{OUT}/fig_ss_folded_structure.png/.pdf")
