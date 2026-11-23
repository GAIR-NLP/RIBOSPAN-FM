#!/usr/bin/env python3
"""Faithful recreation of the reference "evolutionary landscape" panel,
matching its actual proportions: a wide, short strip; sequences compact and
tightly stacked in a narrow left column; landscape takes most of the width."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from _paths import REPO

np.random.seed(7)

SEQ_COLOR = "#E8823A"

WT = "GGGUUGGUGGUCUAGCCAGGUUAU"
mut1_pos = [8]
mut2_pos = [8, 16]
ALT = {"G": "C", "U": "A", "C": "G", "A": "U"}

def mutate(seq, positions):
    s = list(seq)
    for p in positions:
        s[p] = ALT[s[p]]
    return "".join(s)

sequences = [WT, mutate(WT, mut1_pos), mutate(WT, mut2_pos)]

fig = plt.figure(figsize=(9.5, 4.4), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[0.62, 1.0], wspace=0.02)

fig.text(0.005, 0.97, "A", fontsize=17, fontweight="bold", color="#111111", va="top")

# ------------------------------------------------------------ Left: sequences
axL = fig.add_subplot(gs[0])
axL.set_xlim(0, 9.0)
axL.set_ylim(0, 6)
axL.axis("off")

mut_positions_list = [set(), set(mut1_pos), set(mut2_pos)]
row_y = [4.55, 3.0, 1.45]
for seq, y, muts in zip(sequences, row_y, mut_positions_list):
    for i, ch in enumerate(seq):
        is_mut = i in muts
        axL.text(0.1 + i * 0.32, y, ch, fontsize=13, fontweight="bold",
                  color=SEQ_COLOR, ha="center", va="center", family="monospace")
        if is_mut:
            axL.plot([0.1 + i * 0.32], [y - 0.5], marker="o", markersize=4.5,
                      color=SEQ_COLOR, clip_on=False)

axL.annotate("", xy=(8.95, 3.0), xytext=(7.9, 3.0), xycoords="data",
             arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))
axL.text(8.42, 3.20, "LIMBER", fontsize=10.5, color="#222222", ha="center", va="bottom")

# ------------------------------------------------------------ Right: 3D landscape
axR = fig.add_subplot(gs[1], projection="3d")

res = 90
x = np.linspace(-3, 3, res)
y = np.linspace(-3, 3, res)
X, Y = np.meshgrid(x, y)
peaks = [((-1.7, -0.9), 0.55, 1.6), ((0.1, 0.7), 0.85, 1.3), ((1.8, -0.2), 1.15, 1.15)]
Z = np.zeros_like(X)
for (px, py), h, w in peaks:
    Z += h * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / w)

# add rugged texture on top of the 3 main peaks (kept dominant so WT/1x/2x
# stay the clear global high points): several secondary peaks, several
# noticeable basins, plus fine random bumps/dips and ripples.
rng = np.random.default_rng(3)

secondary_peaks = [(-2.4, 1.6, 0.72, 0.35), (2.4, 1.8, 0.65, 0.3), (-0.9, -2.2, 0.78, 0.4),
                    (2.6, -1.8, 0.6, 0.3), (-2.6, -1.6, 0.68, 0.35), (0.9, -1.3, 0.52, 0.25)]
for bx, by, bh, bw in secondary_peaks:
    Z += bh * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / bw)

basins = [(-0.8, 1.9, -0.7, 0.5), (1.0, 2.2, -0.62, 0.45), (-2.0, -0.1, -0.68, 0.4),
          (0.9, -0.4, -0.55, 0.35), (2.3, 0.9, -0.55, 0.35), (-1.0, -2.0, -0.58, 0.4)]
for bx, by, bh, bw in basins:
    Z += bh * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / bw)

for _ in range(24):
    bx, by = rng.uniform(-3, 3, 2)
    bh = rng.uniform(-0.22, 0.26)
    bw = rng.uniform(0.1, 0.3)
    Z += bh * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / bw)
Z += 0.08 * np.sin(X * 4.5) * np.cos(Y * 3.7)
Z += 0.05 * np.sin(X * 8.0 + 1.0) * np.sin(Y * 7.0)

cmap = LinearSegmentedColormap.from_list("fitness", ["#3D6FA6", "#EDF1F5", "#C0392B"])
axR.plot_surface(X, Y, Z, cmap=cmap, linewidth=0.4, edgecolor="#FFFFFF",
                  antialiased=True, rstride=1, cstride=1, alpha=0.98)

marker_pts = [(-1.7, -0.9, "WT"), (0.1, 0.7, "1x"), (1.8, -0.2, "2x")]
zmin, zmax = Z.min(), Z.max()
for px, py, label in marker_pts:
    i = np.argmin(np.abs(x - px)); j = np.argmin(np.abs(y - py))
    pz = Z[j, i]
    c = cmap((pz - zmin) / (zmax - zmin))
    axR.plot([px], [py], [pz + 0.35], marker="o", markersize=9, markerfacecolor=c,
              markeredgecolor="white", markeredgewidth=1.6, zorder=100)
    axR.text(px, py, pz + 0.75, label, fontsize=10.5, fontweight="bold",
              color="#1A1A1A", ha="center", va="bottom", zorder=101)

axR.set_box_aspect((1, 1, 0.62))
axR.set_axis_off()
axR.view_init(elev=27, azim=-55)
axR.text2D(0.5, 0.92, "Fitness landscape prediction", transform=axR.transAxes,
           ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A1A")

cax = fig.add_axes([0.74, 0.12, 0.14, 0.035])
grad = np.linspace(0, 1, 256).reshape(1, -1)
cax.imshow(grad, aspect="auto", cmap=cmap)
cax.set_xticks([]); cax.set_yticks([])
for spine in cax.spines.values():
    spine.set_visible(False)
cax.text(0, -1.5, "low", transform=cax.transAxes, ha="left", va="top", fontsize=9, color="#555555")
cax.text(1, -1.5, "high", transform=cax.transAxes, ha="right", va="top", fontsize=9, color="#555555")

fig.savefig(str(REPO / "outputs/fig_fitness_landscape.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(str(REPO / "outputs/fig_fitness_landscape.pdf"),
            dpi=300, bbox_inches="tight", facecolor="white")
print("saved fig_fitness_landscape.png/.pdf")
