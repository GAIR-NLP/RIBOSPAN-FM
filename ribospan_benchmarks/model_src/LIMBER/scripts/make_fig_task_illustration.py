#!/usr/bin/env python3
"""Illustrate the two downstream TASKS themselves (not just performance
numbers): IRES-mediated cap-independent translation initiation, and mean
ribosome load (how many ribosomes a 5'UTR recruits). Real example sequences
+ real labels/measurements, drawn as nucleotide-color strips with simple
ribosome pictograms."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrow, FancyBboxPatch
from _paths import REPO

NT_COLOR = {"A": "#2ECC71", "U": "#E74C3C", "G": "#F1C40F", "C": "#3498DB"}
RIBO = "#6C5CE7"
INK = "#111111"

OUT = str(REPO / "outputs")

# ------------------------------------------------------------------ data
ires_pos = "GUGGCUGAGACUCAACUGUCACCCCCUCCUCUGGCUCCAUCCCUUCCGUCCCCUUUUGCCUCUUUCUCUUUUUGGUCGGGGACAGGACUGUGUCUUCUAGAGGCUCAGUGUUAAUCUGUU"
ires_neg = "AGCUCUCAGGGGUCGGCGGGUGACUUCUUUCCGGAAGAAAGCGAGGAACGCGCUCUGCGGGGUGAGCCGGACUCCCCAACUCCGGACGAUCAGCCCAGGACUGAGAGCCCCGAAGUCCCC"
ires_pos_p, ires_neg_p = 0.99998, 8.4e-7

mrl_examples = [
    ("UUACAAUAAAAUAGACGACGCCAGCCCAGAUCACGAUGUAAGGAACAAUUUAAGAU", 2.91),
    ("UCGGACCUUUCGUCGGAGUACUCCGAC", 5.36),
    ("CACUUGCAGCAAGGCCAGACCUGUUGCCAUAUCGUGCAACA", 6.07),
    ("CGAUAAAACGUCAAAGAGCGUAGUAACGACGUCUCAUGAUCAACCAAAAU", 6.64),
]

def draw_strip(ax, seq, x0, y0, cell_w, cell_h):
    for i, b in enumerate(seq):
        ax.add_patch(Rectangle((x0 + i * cell_w, y0), cell_w * 0.92, cell_h,
                                 facecolor=NT_COLOR[b], edgecolor="none", zorder=2))
    return x0 + len(seq) * cell_w

fig = plt.figure(figsize=(11, 11), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.55)

# ================================================================== Panel A: IRES
axA = fig.add_subplot(gs[0])
axA.set_xlim(0, 140)
axA.set_ylim(0, 10)
axA.axis("off")

cw = 118 / len(ires_pos)

# positive example (row at y=6)
y = 6.4
draw_strip(axA, ires_pos, 0, y, cw, 1.1)
# cap symbol crossed out (no scanning) + internal ribosome landing arrow
cap_x = 0
axA.add_patch(Circle((cap_x - 2.2, y + 0.55), 0.55, facecolor="#DDDDDD", edgecolor=INK, lw=1.2, zorder=3))
axA.plot([cap_x - 2.8, cap_x - 1.6], [y + 0.05, y + 1.05], color="#E74C3C", lw=2.2, zorder=4)
axA.plot([cap_x - 2.8, cap_x - 1.6], [y + 1.05, y + 0.05], color="#E74C3C", lw=2.2, zorder=4)
land_x = 0 + 0.62 * len(ires_pos) * cw
axA.annotate("", xy=(land_x, y + 1.15), xytext=(land_x, y + 2.6),
             arrowprops=dict(arrowstyle="-|>", color=RIBO, lw=3))
axA.add_patch(Circle((land_x, y + 1.15), 0.62, facecolor=RIBO, edgecolor="white", lw=1.5, zorder=5))
axA.text(land_x, y + 3.0, "ribosome lands\ndirectly on IRES", ha="center", va="bottom",
          fontsize=10.5, color=RIBO, fontweight="bold")
axA.text(-4, y + 0.55, "IRES +", ha="right", va="center", fontsize=15, fontweight="bold", color="#1B8A4A")
badge = FancyBboxPatch((120, y - 0.15), 20, 1.35, boxstyle="round,pad=0.02,rounding_size=0.25",
                        facecolor="#1B8A4A", edgecolor="none", zorder=6)
axA.add_patch(badge)
axA.text(130, y + 0.5, f"P={ires_pos_p:.4f}", ha="center", va="center", fontsize=12.5,
          fontweight="bold", color="white", zorder=7)

# negative example (row at y=1.0)
y = 1.0
draw_strip(axA, ires_neg, 0, y, cw, 1.1)
axA.add_patch(Circle((cap_x - 2.2, y + 0.55), 0.55, facecolor="#F1C40F", edgecolor=INK, lw=1.2, zorder=3))
axA.annotate("", xy=(0, y + 0.55), xytext=(-1.6, y + 0.55),
             arrowprops=dict(arrowstyle="-|>", color="#999999", lw=2.5))
axA.text(-4, y + 0.55, "IRES −", ha="right", va="center", fontsize=15, fontweight="bold", color="#B0413E")
axA.text((0 + 118) / 2, y - 1.1, "no internal entry — ribosome only enters by scanning from the cap",
          ha="center", va="top", fontsize=10.5, color="#777777", style="italic")
badge2 = FancyBboxPatch((120, y - 0.15), 20, 1.35, boxstyle="round,pad=0.02,rounding_size=0.25",
                         facecolor="#B0413E", edgecolor="none", zorder=6)
axA.add_patch(badge2)
axA.text(130, y + 0.5, f"P={ires_neg_p:.1e}", ha="center", va="center", fontsize=12.5,
          fontweight="bold", color="white", zorder=7)

# ================================================================== Panel B: MRL
axB = fig.add_subplot(gs[1])
axB.set_xlim(0, 140)
axB.set_ylim(0, 10.5)
axB.axis("off")

row_y = np.linspace(8.6, 0.8, len(mrl_examples))
max_len = max(len(s) for s, _ in mrl_examples)
for (seq, rl), y in zip(mrl_examples, row_y):
    cw2 = 80 / max_len
    draw_strip(axB, seq, 0, y, cw2, 0.9)
    n_ribo = max(1, round(rl))
    for k in range(n_ribo):
        rx = 92 + k * 2.3
        axB.add_patch(Circle((rx, y + 0.45), 0.85, facecolor=RIBO,
                               edgecolor="white", lw=1.3, alpha=0.55 + 0.45 * (k == n_ribo - 1), zorder=3))
    axB.text(139, y + 0.45, f"RL={rl:.2f}", ha="right", va="center", fontsize=13,
              fontweight="bold", color=INK)

axB.text(46, 9.9, "5' UTR sequence", ha="center", fontsize=11, color="#777777", style="italic")
axB.text(104, 9.9, "ribosomes loaded", ha="center", fontsize=11, color="#777777", style="italic")

fig.suptitle("LIMBER downstream tasks, in biological terms", fontsize=20, fontweight="bold", y=1.01)

fig.savefig(f"{OUT}/fig_task_illustration.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_task_illustration.pdf", dpi=300, bbox_inches="tight", facecolor="white")
print("saved:", f"{OUT}/fig_task_illustration.png/.pdf")
