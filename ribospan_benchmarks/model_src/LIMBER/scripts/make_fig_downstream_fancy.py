#!/usr/bin/env python3
"""Fancy downstream-adaptation dashboard: IRES classification (radar, AUPRC
across 5 held-out test regimes) + MRL regression (R^2 vs 5'UTR length bin).
Compares LIMBER-650M against general-purpose giants (RiNALMo-giga,
AIDO.RNA-650M) and the task-specific IRES SOTA (DeepIRES)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from _paths import REPO

OUT = str(REPO / "outputs")

COLORS = {
    "LIMBER-650M":   "#E63946",
    "RiNALMo-giga":  "#457B9D",
    "AIDO.RNA-650M": "#F4A261",
    "DeepIRES":      "#6B6B6B",
}
LW = {"LIMBER-650M": 4.5, "RiNALMo-giga": 3, "AIDO.RNA-650M": 3, "DeepIRES": 2.5}
LS = {"LIMBER-650M": "-", "RiNALMo-giga": "-", "AIDO.RNA-650M": "-", "DeepIRES": "--"}
ZORDER = {"LIMBER-650M": 5, "RiNALMo-giga": 3, "AIDO.RNA-650M": 3, "DeepIRES": 2}

plt.rcParams["font.weight"] = "medium"

fig = plt.figure(figsize=(16, 7.5), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.32)

# ---------------------------------------------------------------- Panel A: IRES radar
ires = pd.read_csv(f"{OUT}/ires_results_all_models.csv")
test_sets = ["test_bal", "test_imbal", "test_main", "test_core", "test_5utr"]
labels = ["Balanced", "Imbalanced", "Main", "Core", "5'UTR"]
models_a = ["LIMBER-650M", "RiNALMo-giga", "AIDO.RNA-650M", "DeepIRES"]

n = len(test_sets)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles += angles[:1]

ax1 = fig.add_subplot(gs[0], projection="polar")
ax1.set_theta_offset(np.pi / 2)
ax1.set_theta_direction(-1)
ax1.set_facecolor("#FAFAFA")

for model in models_a:
    sub = ires[ires["model"] == model].set_index("test_set")
    vals = [sub.loc[ts, "auprc"] for ts in test_sets]
    vals += vals[:1]
    ax1.plot(angles, vals, color=COLORS[model], lw=LW[model], linestyle=LS[model],
              zorder=ZORDER[model], solid_capstyle="round", label=model)
    fill_alpha = 0.22 if model == "LIMBER-650M" else 0.0
    if fill_alpha:
        ax1.fill(angles, vals, color=COLORS[model], alpha=fill_alpha, zorder=1)
    ax1.scatter(angles[:-1], vals[:-1], s=70 if model == "LIMBER-650M" else 40,
                color=COLORS[model], edgecolor="white", linewidth=1.5, zorder=ZORDER[model] + 1)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(labels, fontsize=15, fontweight="bold")
ax1.set_ylim(0, 1)
ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
ax1.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=10, color="#888888")
ax1.spines["polar"].set_color("#CCCCCC")
ax1.grid(color="#DDDDDD", linewidth=1)
ax1.set_title("IRES detection — AUPRC across held-out regimes", fontsize=16,
               fontweight="bold", pad=28, color="#222222")

# ---------------------------------------------------------------- Panel B: MRL by length bin
mrl = pd.read_csv(f"{OUT}/mrl_results_all_models.csv")
bins = ["25-44", "45-64", "65-84", "85-100"]
bin_x = np.arange(len(bins))
models_b = ["LIMBER-650M", "RiNALMo-giga", "AIDO.RNA-650M"]

ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#FAFAFA")

for model in models_b:
    sub = mrl[(mrl["model"] == model) & (mrl["test_set"] == "test_random") & (mrl["length_bin"] != "overall")]
    sub = sub.set_index("length_bin").loc[bins]
    y = sub["r2"].to_numpy()
    ax2.plot(bin_x, y, color=COLORS[model], lw=LW[model] + 1.5, linestyle=LS[model],
              zorder=ZORDER[model], solid_capstyle="round", marker="o",
              markersize=13 if model == "LIMBER-650M" else 9,
              markerfacecolor=COLORS[model], markeredgecolor="white", markeredgewidth=2,
              label=model)
    if model == "LIMBER-650M":
        ax2.fill_between(bin_x, y, y.min() - 0.06, color=COLORS[model], alpha=0.12, zorder=0)

ax2.set_xticks(bin_x)
ax2.set_xticklabels([f"{b} nt" for b in bins], fontsize=14, fontweight="bold")
ax2.set_ylabel("R² (test_random)", fontsize=14, fontweight="bold")
ax2.set_ylim(0.78, 0.95)
ax2.tick_params(axis="y", labelsize=12)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax2.spines[spine].set_color("#999999")
ax2.grid(axis="y", color="#E0E0E0", linewidth=1, zorder=0)
ax2.set_title("Mean ribosome load — R² vs 5'UTR length", fontsize=16,
               fontweight="bold", pad=18, color="#222222")

# ---------------------------------------------------------------- shared legend
handles = [plt.Line2D([0], [0], color=COLORS[m], lw=4, linestyle=LS[m], label=m)
           for m in ["LIMBER-650M", "RiNALMo-giga", "AIDO.RNA-650M", "DeepIRES"]]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13.5,
           frameon=False, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("LIMBER downstream adaptation: IRES detection & translation-efficiency prediction",
             fontsize=19, fontweight="bold", y=1.03, color="#111111")

fig.savefig(f"{OUT}/fig_downstream_fancy.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_downstream_fancy.pdf", bbox_inches="tight", facecolor="white")
print("saved:", f"{OUT}/fig_downstream_fancy.png/.pdf")
