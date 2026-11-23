#!/usr/bin/env python3
"""8-model comparison violin: x = 9 RNA families, violins per family (LFO top, Within bottom)."""
import json, numpy as np, matplotlib, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from pathlib import Path
import statistics
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

BASE = REPO / "outputs"
ALL_FAMS = ['5s','16s','23s','grp1','srp','telomerase','RNaseP','tmRNA','tRNA']
LBLS = {'telomerase':'Telomerase','grp1':'Group I','srp':'SRP RNA','16s':'16S rRNA',
        'RNaseP':'RNase P','tmRNA':'tmRNA','23s':'23S rRNA','5s':'5S rRNA','tRNA':'tRNA'}

# ── Model definitions ─────────────────────────────────────────────────────────
# (display_name, color, lfo_dir, within_dir, is_ours)
MODELS = [
    ('LIMBER-100M',   '#1565C0', BASE/'fm_hybrid_100m/ss_eval_perseq/epoch12_lfo',    BASE/'fm_hybrid_100m/ss_eval_perseq/epoch12_within',   True),
    ('LIMBER-300M',   '#0D47A1', BASE/'fm_hybrid_300m/ss_eval_perseq/epoch12_lfo',    BASE/'fm_hybrid_300m/ss_eval_perseq/epoch12_within',    True),
    ('LIMBER-650M',   '#1976D2', BASE/'fm_hybrid_650m/ss_eval_perseq/epoch12_lfo',    BASE/'fm_hybrid_650m/ss_eval_perseq/epoch12_within',    True),
    ('Transformer-100M', '#C0392B', BASE/'bert_100m/ss_eval_perseq/epoch6_lfo',       BASE/'bert_100m/ss_eval_perseq/epoch6_within',          False),
    ('RNA-FM',        '#D4760A', BASE/'rnafm_100m/ss_eval_perseq/lfo',                BASE/'rnafm_100m/ss_eval_perseq/within',                False),
    ('RiNALMo-giga',  '#7B1FA2', BASE/'rinalmo_giga/ss_eval_perseq/lfo',              BASE/'rinalmo_giga/ss_eval_perseq/within',              False),
    ('AIDO.RNA-650M', '#00796B', BASE/'aido_rna_650m/ss_eval_perseq/lfo',             BASE/'aido_rna_650m/ss_eval_perseq/within',             False),
    ('StructRFM',    '#558B2F', BASE/'structrfm_86m/ss_eval_perseq/lfo',              BASE/'structrfm_86m/ss_eval_perseq/within',              False),
]

N_MOD  = len(MODELS)
SEP    = 0.112          # uniform inter-model spacing
HW     = 0.046          # violin max half-width
BW_BOX = 0.020          # IQR box half-width
OFFSETS = (np.arange(N_MOD) - (N_MOD - 1) / 2) * SEP   # -0.392 … +0.392

def load_dir(d):
    return {f: json.load(open(Path(d)/f'{f}.json'))
            for f in ALL_FAMS if (Path(d)/f'{f}.json').exists()}

def full_kde(arr, hw):
    a = np.asarray(arr, float)
    if len(a) < 3 or np.std(a) < 1e-8:
        m = float(a.mean())
        return np.array([-0.04, m, 1.05]), np.array([0., hw * .55, 0.])
    kf = gaussian_kde(a, bw_method='silverman')
    xs = np.linspace(-0.04, 1.05, 350)
    d  = kf(xs); d = d / d.max() * hw
    return xs, d

MODEL_DATA = [(n, c, load_dir(ld), load_dir(wd), o) for n, c, ld, wd, o in MODELS]

# Sort families by LIMBER-650M LFO mean_F1 ascending
ref = MODEL_DATA[2][2]   # LIMBER-650M LFO
FAMS = sorted(ALL_FAMS, key=lambda f: ref[f]['mean_F1'] if f in ref else 0)
N_FAM = len(FAMS)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(22, 9.5), sharex=True)
rng = np.random.default_rng(42)

for row, (ax, is_lfo) in enumerate(zip(axes, [True, False])):

    # Alternating family background for readability
    for i in range(N_FAM):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='#F7F7F7', alpha=0.7, zorder=0, lw=0)

    for i, fam in enumerate(FAMS):
        for mi, (name, clr, lfo_d, wth_d, ours) in enumerate(MODEL_DATA):
            d   = lfo_d if is_lfo else wth_d
            if fam not in d: continue
            arr = np.asarray(d[fam]['per_seq_f1'], float)
            cx  = i + OFFSETS[mi]

            # Violin
            ys, ds = full_kde(arr, HW)
            ax.fill_betweenx(ys, cx - ds, cx + ds, color=clr, alpha=0.20, lw=0, zorder=2)
            ax.plot(cx - ds, ys, color=clr, alpha=0.68, lw=0.85, zorder=2)
            ax.plot(cx + ds, ys, color=clr, alpha=0.68, lw=0.85, zorder=2)

            # IQR box + median
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            ax.add_patch(mpatches.Rectangle(
                (cx - BW_BOX, q1), 2*BW_BOX, q3 - q1,
                fc=clr, ec='none', alpha=0.78, zorder=3))
            ax.plot([cx - BW_BOX, cx + BW_BOX], [med, med],
                    color='white', lw=1.5, zorder=4)
            # mean diamond
            ax.scatter([cx], [float(arr.mean())], marker='D', s=16,
                       fc='white', ec=clr, lw=0.75, zorder=5)

        # n= label (from first model)
        d0 = MODEL_DATA[0][2 if is_lfo else 3]
        n_test = d0[fam]['num_test'] if fam in d0 else '?'
        ax.text(i, -0.095, f'n={n_test}', ha='center', va='top',
                fontsize=8.5, color='#666')

    ax.set_ylim(-0.13, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='y', labelsize=11)
    ax.yaxis.grid(True, linestyle='-', lw=0.4, color='black', alpha=0.09)
    ax.set_axisbelow(True)
    for sp in ['top', 'right', 'bottom']: ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    split_lbl = 'LFO' if is_lfo else 'Within-family'
    ax.set_ylabel(f'Flexible F1 (±1 nt)\n[{split_lbl}]', fontsize=12.5, labelpad=6)

    # Macro F1 label for each model in top-right corner
    if row == 0:
        macro_strs = []
        for mi, (name, clr, lfo_d, wth_d, ours) in enumerate(MODEL_DATA):
            vals = [lfo_d[f]['mean_F1'] for f in FAMS if f in lfo_d]
            m = statistics.mean(vals) if vals else 0
            macro_strs.append(f'{m:.3f}')

# x-axis ticks
axes[1].set_xticks(range(N_FAM))
axes[1].set_xticklabels([LBLS[f] for f in FAMS], fontsize=11.5,
                          color='black', fontweight='bold')
axes[1].tick_params(axis='x', length=0, pad=13)
axes[1].set_xlim(-0.60, N_FAM - 0.42)

plt.subplots_adjust(hspace=0.08, bottom=0.13)

# ── Legend ─────────────────────────────────────────────────────────────────────
handles, labels = [], []
for mi, (name, clr, lfo_d, wth_d, ours) in enumerate(MODEL_DATA):
    patch = mpatches.Patch(fc=clr, alpha=0.78, ec='none')
    lfo_m = statistics.mean([lfo_d[f]['mean_F1'] for f in FAMS if f in lfo_d])
    wth_m = statistics.mean([wth_d[f]['mean_F1'] for f in FAMS if f in wth_d])
    star  = '★ ' if ours else ''
    handles.append(patch)
    labels.append(f'{star}{name}   LFO={lfo_m:.3f}  Within={wth_m:.3f}')

fig.legend(handles, labels, frameon=False, fontsize=9.8,
           loc='lower center', bbox_to_anchor=(0.5, 0.022),
           ncol=4, handlelength=1.3, handletextpad=0.45, columnspacing=1.2)

out = BASE / 'violin_compare_all'
fig.savefig(str(out)+'.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(str(out)+'.pdf', bbox_inches='tight', facecolor='white')
print(f'Saved: {out}.pdf / .png')
