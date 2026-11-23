#!/usr/bin/env python3
"""Layer-importance strip (T/M tiles, LIMBER-650M, epoch6): color saturation
encodes how much zero-shot SS quality (macro NormAUPR) drops when that layer
is ablated — brighter red = more important. Middle (flat, low-importance
layers) collapsed to "......", mirroring the reference mock-up."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from _paths import REPO

OUT = str(REPO / "outputs")
PATTERN = "TTMTTMTTMTTMTTMTTMTTMTTMTTMTTMTTM"

base = json.load(open(f"{OUT}/jacobian_eval/hybrid_650m_ep6/baseline_summary.json"))
base_macro = np.mean([v["mean_NormAUPR"] for v in base.values()])
abl = json.load(open(f"{OUT}/jacobian_eval/hybrid_650m_ep6/ablation_summary.json"))

importance = []
for k, lt in enumerate(PATTERN):
    tag = f"layer_{k:02d}_{lt}"
    macro = np.mean([v["mean_NormAUPR"] for v in abl[tag].values()])
    importance.append(max(base_macro - macro, 0.0))
importance = np.array(importance)
norm_imp = importance / importance.max()

PREFIX = list(range(0, 9))     # layers 0-8   (includes early spikes at 2,5,8)
SUFFIX = list(range(28, 33))   # layers 28-32 (includes the big spike at 29)

cmap = LinearSegmentedColormap.from_list("imp", ["#FDE1E3", "#FF1744"])

fig, ax = plt.subplots(figsize=(16, 2.4))
tile_w, gap, x = 1.0, 0.18, 0.0

def draw_tile(x, idx):
    color = cmap(norm_imp[idx])
    box = FancyBboxPatch((x, 0), tile_w, tile_w, boxstyle="round,pad=0,rounding_size=0.14",
                          linewidth=0, facecolor=color, zorder=2)
    ax.add_patch(box)
    ax.text(x + tile_w / 2, tile_w / 2, PATTERN[idx], ha="center", va="center",
            fontsize=30, fontweight="bold", color="#1A1A1A", zorder=3)

for idx in PREFIX:
    draw_tile(x, idx)
    x += tile_w + gap

ax.text(x + 0.55, tile_w / 2, "......", ha="center", va="center",
        fontsize=30, fontweight="bold", color="#333333", zorder=3)
x += 1.5

for idx in SUFFIX:
    draw_tile(x, idx)
    x += tile_w + gap

ax.set_xlim(-0.2, x)
ax.set_ylim(-0.15, tile_w + 0.15)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig_layer_importance_strip.png", dpi=300, transparent=True)
fig.savefig(f"{OUT}/fig_layer_importance_strip.pdf", dpi=300)
print("saved:", f"{OUT}/fig_layer_importance_strip.png/.pdf")
print("importance (raw drop) by layer:", dict(enumerate(np.round(importance, 4))))
