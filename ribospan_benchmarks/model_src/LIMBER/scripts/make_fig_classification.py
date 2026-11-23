#!/usr/bin/env python3
"""Classification-figure: LDA scatter of LIMBER-650M CLS embeddings colored by
ArchiveII family, using the full (uncapped) per-family sequence set."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from _paths import REPO

OUT = str(REPO / "outputs")

X = np.load(f"{OUT}/family_embeddings_cls_full.npy")
labels = np.array(json.load(open(f"{OUT}/family_embeddings_cls_full_labels.json")))

Xs = StandardScaler().fit_transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
Z = lda.fit_transform(Xs, labels)

# jitter scale fixed from the RAW (pre-compression) spread — must NOT be
# recomputed after compression, or it shrinks in lockstep with the gaps and
# the compression below becomes a visual no-op (self-similar rescaling).
rng = np.random.default_rng(0)
jitter_scale = 0.028 * (Z[:, 0].max() - Z[:, 0].min())
jitter = rng.normal(0, jitter_scale, size=Z.shape)

# Pull cluster centroids toward the global center to shrink inter-cluster gaps
# while leaving each cluster's internal (within-family) spread untouched.
SHRINK = 0.38
global_mean = Z.mean(axis=0)
Z_compressed = np.empty_like(Z)
for fam in np.unique(labels):
    m = labels == fam
    centroid = Z[m].mean(axis=0)
    new_centroid = global_mean + SHRINK * (centroid - global_mean)
    Z_compressed[m] = new_centroid + (Z[m] - centroid)
Z = Z_compressed + jitter

families = ["tRNA", "5s", "16s", "23s", "RNaseP", "srp", "grp1", "telomerase", "tmRNA"]
colors = {
    "tRNA":       "#E53935",  # vivid red
    "5s":         "#1E88E5",  # vivid blue
    "16s":        "#43A047",  # vivid green
    "23s":        "#FDD835",  # vivid yellow
    "RNaseP":     "#8E24AA",  # vivid purple
    "srp":        "#FB8C00",  # vivid orange
    "grp1":       "#00ACC1",  # vivid cyan
    "telomerase": "#D81B60",  # vivid pink
    "tmRNA":      "#3949AB",  # vivid indigo
}

fig, ax = plt.subplots(figsize=(8, 8))
for fam in families:
    m = labels == fam
    ax.scatter(Z[m, 0], Z[m, 1],
               s=90, color=colors[fam], alpha=0.75,
               edgecolor="white", linewidth=0.6, zorder=2)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_aspect("auto")

# Fix view limits from a robust (percentile-based) box so a few stray outlier
# points don't stretch the frame and silently undo the inter-cluster shrink.
lo_x, hi_x = np.percentile(Z[:, 0], [1, 99])
lo_y, hi_y = np.percentile(Z[:, 1], [1, 99])
pad_x = 0.12 * (hi_x - lo_x)
pad_y = 0.12 * (hi_y - lo_y)
ax.set_xlim(lo_x - pad_x, hi_x + pad_x)
ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
fig.tight_layout(pad=0.5)
fig.savefig(f"{OUT}/fig_classification_lda.png", dpi=300, transparent=True)
fig.savefig(f"{OUT}/fig_classification_lda.pdf", dpi=300)
print("saved:", f"{OUT}/fig_classification_lda.png/.pdf")
print("explained_variance_ratio:", lda.explained_variance_ratio_)
print("n per family:", {f: int((labels == f).sum()) for f in families})
