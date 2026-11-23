#!/usr/bin/env python3
"""Downstream-capability showcase for LIMBER-650M (no baseline comparison —
just "what LIMBER can do"): IRES detection ROC curve + MRL prediction quality
across 5'UTR length. Fancy, bold, gradient-filled, big stat badges."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, auc
from _paths import REPO

OUT = str(REPO / "outputs")
RED = "#E63946"
NAVY = "#1D3557"

fig = plt.figure(figsize=(16, 7.2), facecolor="white")
gs = fig.add_gridspec(1, 2, wspace=0.3)

# ---------------------------------------------------------------- Panel A: IRES ROC
preds = json.load(open(f"{OUT}/downstream_ft/ires/fm_hybrid_650m/predictions.json"))
tb = preds["test_bal"]
y_true = np.array(tb["y_true"])
y_prob = np.array(tb["y_prob"])
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#FAFAFA")
ax1.plot([0, 1], [0, 1], color="#BBBBBB", lw=2.5, linestyle="--", zorder=1)
ax1.plot(fpr, tpr, color=RED, lw=6, zorder=3, solid_capstyle="round")
ax1.fill_between(fpr, tpr, 0, color=RED, alpha=0.15, zorder=2)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)
ax1.set_xlabel("False positive rate", fontsize=15, fontweight="bold")
ax1.set_ylabel("True positive rate", fontsize=15, fontweight="bold")
ax1.tick_params(labelsize=12)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax1.spines[spine].set_color("#999999")
ax1.grid(color="#E5E5E5", lw=1, zorder=0)
ax1.set_title("IRES detection", fontsize=19, fontweight="bold", pad=16, color="#111111")

badge = FancyBboxPatch((0.42, 0.06), 0.52, 0.20, transform=ax1.transAxes,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=RED, edgecolor="none", zorder=5, alpha=0.95)
ax1.add_patch(badge)
ax1.text(0.68, 0.16, f"AUROC {roc_auc:.2f}", transform=ax1.transAxes,
          ha="center", va="center", fontsize=22, fontweight="bold", color="white", zorder=6)

# ---------------------------------------------------------------- Panel B: MRL by length bin
mrl = json.load(open(f"{OUT}/downstream_ft/mrl/fm_hybrid_650m/results.json"))["results"]["test_random"]
bins = ["25-44", "45-64", "65-84", "85-100"]
bin_x = np.arange(len(bins))
r2 = np.array([mrl["by_length_bin"][b]["r2"] for b in bins])
overall_r2 = mrl["r2"]

ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#FAFAFA")
YLO = 0.70
ax2.plot(bin_x, r2, color=NAVY, lw=6, zorder=3, solid_capstyle="round", marker="o",
          markersize=16, markerfacecolor=NAVY, markeredgecolor="white", markeredgewidth=3)
ax2.fill_between(bin_x, r2, YLO, color=NAVY, alpha=0.15, zorder=2)

ax2.set_xticks(bin_x)
ax2.set_xticklabels([f"{b} nt" for b in bins], fontsize=14, fontweight="bold")
ax2.set_ylabel("R²  (mean ribosome load)", fontsize=15, fontweight="bold")
ax2.set_ylim(YLO, 0.98)
ax2.tick_params(axis="y", labelsize=12)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax2.spines[spine].set_color("#999999")
ax2.grid(axis="y", color="#E5E5E5", lw=1, zorder=0)
ax2.set_title("Translation-efficiency prediction", fontsize=19, fontweight="bold", pad=16, color="#111111")

badge2 = FancyBboxPatch((0.30, 0.06), 0.66, 0.15, transform=ax2.transAxes,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=NAVY, edgecolor="none", zorder=5, alpha=0.95)
ax2.add_patch(badge2)
ax2.text(0.63, 0.135, f"R² {overall_r2:.2f} overall", transform=ax2.transAxes,
          ha="center", va="center", fontsize=20, fontweight="bold", color="white", zorder=6)

fig.suptitle("LIMBER adapts to downstream RNA function prediction",
             fontsize=21, fontweight="bold", y=1.02, color="#111111")

fig.savefig(f"{OUT}/fig_downstream_capability.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_downstream_capability.pdf", bbox_inches="tight", facecolor="white")
print("saved:", f"{OUT}/fig_downstream_capability.png/.pdf")
print("AUROC:", roc_auc, "overall R2:", overall_r2)
