#!/usr/bin/env python3
"""Grouped bar comparison: IRES classification (AUROC) + MRL regression (R2),
across all 7 backbones. Companion to plot_violin_compare_all.py (SS task)."""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from _paths import REPO

matplotlib.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11, 'text.color': 'black',
    'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    'axes.linewidth': 0.6, 'ytick.major.width': 0.6, 'ytick.major.size': 3.5,
    'xtick.major.size': 0,
})

BASE = REPO / "outputs/downstream_ft"
OUT  = REPO / "outputs"

MODELS = [
    ('LIMBER-100M',   '#1565C0', 'fm_hybrid_100m', True),
    ('LIMBER-300M',   '#0D47A1', 'fm_hybrid_300m', True),
    ('LIMBER-650M',   '#1976D2', 'fm_hybrid_650m', True),
    ('RiNALMo-giga',  '#7B1FA2', 'rinalmo_giga',   False),
    ('AIDO.RNA-650M', '#00796B', 'aido_rna_650m',  False),
    ('RNA-FM',        '#D4760A', 'rnafm_100m',     False),
    ('StructRFM',     '#558B2F', 'structrfm_86m',  False),
]
N = len(MODELS)

IRES_SETS = [('test_bal', 'Balanced'), ('test_imbal', 'Imbalanced'),
             ('test_main', 'Main (cross-sp.)'), ('test_core', 'Core (cross-sp.)'),
             ('test_5utr', "5'UTR (cross-sp.)")]
MRL_SETS  = [('test_random', 'Random 5\'UTR'), ('test_human', 'Human 5\'UTR')]

ires_data = {name: json.load(open(BASE/'ires'/tag/'results.json'))['results'] for name, _, tag, _ in MODELS}
mrl_data  = {name: json.load(open(BASE/'mrl'/tag/'results.json'))['results']  for name, _, tag, _ in MODELS}

SEP = 0.115
OFFSETS = (np.arange(N) - (N - 1) / 2) * SEP
BW = 0.10

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={'width_ratios': [5, 2]})

# ---- IRES panel ----
for i, (key, lbl) in enumerate(IRES_SETS):
    if i % 2 == 0:
        ax1.axvspan(i - 0.5, i + 0.5, color='#F7F7F7', alpha=0.7, zorder=0, lw=0)
    for mi, (name, clr, tag, ours) in enumerate(MODELS):
        v = ires_data[name][key]['auroc']
        cx = i + OFFSETS[mi]
        ax1.bar(cx, v, width=BW, color=clr, alpha=0.88 if ours else 0.72,
                edgecolor='white', linewidth=0.4, zorder=3)

ax1.axhline(0.5, color='#999', lw=0.9, ls='--', zorder=2)
ax1.text(len(IRES_SETS) - 0.42, 0.5, 'random', fontsize=8.5, color='#777',
          va='bottom', ha='right')
ax1.set_xticks(range(len(IRES_SETS)))
ax1.set_xticklabels([lbl for _, lbl in IRES_SETS], fontsize=11, fontweight='bold')
ax1.set_xlim(-0.62, len(IRES_SETS) - 0.38)
ax1.set_ylim(0, 1.0)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.yaxis.grid(True, linestyle='-', lw=0.4, color='black', alpha=0.09)
ax1.set_axisbelow(True)
for sp in ['top', 'right']: ax1.spines[sp].set_visible(False)
ax1.set_ylabel('AUROC', fontsize=12.5)
ax1.set_title('IRES classification', fontsize=13, fontweight='bold', pad=10)

# ---- MRL panel ----
for i, (key, lbl) in enumerate(MRL_SETS):
    if i % 2 == 0:
        ax2.axvspan(i - 0.5, i + 0.5, color='#F7F7F7', alpha=0.7, zorder=0, lw=0)
    for mi, (name, clr, tag, ours) in enumerate(MODELS):
        v = mrl_data[name][key]['r2']
        cx = i + OFFSETS[mi]
        ax2.bar(cx, v, width=BW, color=clr, alpha=0.88 if ours else 0.72,
                edgecolor='white', linewidth=0.4, zorder=3)

ax2.set_xticks(range(len(MRL_SETS)))
ax2.set_xticklabels([lbl for _, lbl in MRL_SETS], fontsize=11, fontweight='bold')
ax2.set_xlim(-0.62, len(MRL_SETS) - 0.38)
ax2.set_ylim(0, 1.0)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.yaxis.grid(True, linestyle='-', lw=0.4, color='black', alpha=0.09)
ax2.set_axisbelow(True)
for sp in ['top', 'right']: ax2.spines[sp].set_visible(False)
ax2.set_ylabel('R²', fontsize=12.5)
ax2.set_title('5′UTR mean ribosomal loading', fontsize=13, fontweight='bold', pad=10)

# ---- Legend ----
import matplotlib.patches as mpatches
handles = [mpatches.Patch(fc=clr, alpha=0.88 if ours else 0.72, ec='none')
           for _, clr, _, ours in MODELS]
labels  = [('★ ' if ours else '') + name for name, _, _, ours in MODELS]
fig.legend(handles, labels, frameon=False, fontsize=10, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=7, handlelength=1.3,
           handletextpad=0.45, columnspacing=1.3)

plt.subplots_adjust(bottom=0.20, wspace=0.18)

out = OUT / 'ires_mrl_compare'
fig.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(str(out) + '.pdf', bbox_inches='tight', facecolor='white')
print(f'Saved: {out}.pdf / .png')
