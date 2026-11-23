#!/usr/bin/env python3
"""Violin plot: LIMBER-650M secondary structure F1, LFO vs within-family (9 families)."""
import json, numpy as np, matplotlib, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import gaussian_kde
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

CLR_L = '#2171B5'; CLR_W = '#C0392B'
BASE   = REPO / "outputs/fm_hybrid_650m/ss_eval_perseq"
LFO_DIR = BASE / 'epoch12_lfo'
WTH_DIR = BASE / 'epoch12_within'

ALL_FAMS = ['5s','16s','23s','grp1','srp','telomerase','RNaseP','tmRNA','tRNA']
LBLS = {'telomerase':'Telomerase','grp1':'Group I','srp':'SRP RNA','16s':'16S rRNA',
        'RNaseP':'RNase P','tmRNA':'tmRNA','23s':'23S rRNA','5s':'5S rRNA','tRNA':'tRNA'}

LD = {f: json.load(open(LFO_DIR/f'{f}.json')) for f in ALL_FAMS}
WD = {f: json.load(open(WTH_DIR/f'{f}.json')) for f in ALL_FAMS}
FAMS = sorted(ALL_FAMS, key=lambda f: LD[f]['mean_F1'])  # ascending LFO mean
LFO_SHOW_DOTS = {'telomerase', '23s'}  # only small-n families show individual LFO dots

VCOFF = 0.21   # horizontal offset between LFO/within violin centres
HW    = 0.18   # max half-width of each violin
BW    = 0.075  # IQR box half-width

def full_kde(arr, hw):
    """KDE evaluated over [-0.04, 1.05], Silverman bandwidth, normalised to hw."""
    a = np.asarray(arr, float)
    if len(a) < 3:
        m = float(a.mean())
        return np.array([-0.04, m, 1.05]), np.array([0., hw * .6, 0.])
    kf = gaussian_kde(a, bw_method='silverman')
    xs = np.linspace(-0.04, 1.05, 300)
    d  = kf(xs); d = d / d.max() * hw
    return xs, d


fig, ax = plt.subplots(figsize=(12.5, 7.0))
rng = np.random.default_rng(42)
N   = len(FAMS)

for i, fam in enumerate(FAMS):
    x0  = float(i); lcx = x0 - VCOFF; wcx = x0 + VCOFF
    la  = np.asarray(LD[fam]['per_seq_f1'], float)
    wa  = np.asarray(WD[fam]['per_seq_f1'], float)

    # Violin fills + outlines
    for arr, cx, c in [(la, lcx, CLR_L), (wa, wcx, CLR_W)]:
        ys, ds = full_kde(arr, HW)
        ax.fill_betweenx(ys, cx-ds, cx+ds, color=c, alpha=0.20, lw=0, zorder=2)
        ax.plot(cx-ds, ys, color=c, alpha=0.72, lw=0.9, zorder=2)
        ax.plot(cx+ds, ys, color=c, alpha=0.72, lw=0.9, zorder=2)

    # IQR box, median line, mean diamond
    for arr, cx, c in [(la, lcx, CLR_L), (wa, wcx, CLR_W)]:
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        bw = min(HW * 0.55, BW)
        ax.add_patch(mpatches.Rectangle((cx-bw/2, q1), bw, q3-q1,
                                        fc=c, ec='none', alpha=0.72, zorder=3))
        ax.plot([cx-bw/2, cx+bw/2], [med, med], color='white', lw=1.6, zorder=4)
        ax.scatter([cx], [float(arr.mean())], marker='D', s=30,
                   fc='white', ec=c, lw=0.9, zorder=5)

    # Individual data points
    if fam in LFO_SHOW_DOTS:
        jit = rng.uniform(-HW*0.7, HW*0.7, len(la))
        ax.scatter(lcx+jit, la, s=5, color=CLR_L, alpha=0.48, zorder=5, lw=0)
    jit = rng.uniform(-HW*0.7, HW*0.7, len(wa))
    ax.scatter(wcx+jit, wa, s=5, color=CLR_W, alpha=0.48, zorder=5, lw=0)

    # Δ above each family
    delta = WD[fam]['mean_F1'] - LD[fam]['mean_F1']
    ax.text(x0, 1.068, f'{delta:+.3f}', ha='center', va='bottom', fontsize=9.5,
            color='#1A6B35' if delta >= 0.05 else '#555', fontweight='bold')

    # n= labels below x-axis
    nl, nw = LD[fam]['num_test'], WD[fam]['num_test']
    ax.text(lcx, -0.10, f'n={nl}', ha='center', va='top', fontsize=9, color=CLR_L)
    ax.text(wcx, -0.10, f'n={nw}', ha='center', va='top', fontsize=9, color=CLR_W)

# Axes formatting
ax.set_xticks(range(N))
ax.set_xticklabels([LBLS[f] for f in FAMS], fontsize=12, color='black', fontweight='bold')
ax.tick_params(axis='x', length=0, pad=14)
ax.set_ylim(-0.13, 1.14)
ax.set_xlim(-0.5, N - 0.5 + 0.25)
ax.set_ylabel('Flexible F1 (±1 nt)', fontsize=13, labelpad=7)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='y', labelsize=11)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='black', alpha=0.08)
ax.set_axisbelow(True)
for sp in ['top', 'right', 'bottom']: ax.spines[sp].set_visible(False)
ax.spines['left'].set_linewidth(0.6)

# Macro F1 dashed reference lines
ml = np.mean([LD[f]['mean_F1'] for f in FAMS])
mw = np.mean([WD[f]['mean_F1'] for f in FAMS])
for v, c in [(ml, CLR_L), (mw, CLR_W)]:
    ax.axhline(v, ls=(0,(4,4)), lw=0.9, color=c, alpha=0.42, zorder=1)

# Legend: patch + dashed line combined handles; Δ as text-only entry
lfo_patch = mpatches.Patch(fc=CLR_L, alpha=0.72, ec='none')
lfo_dash  = Line2D([0],[0], ls=(0,(3,3)), lw=1.1, color=CLR_L, alpha=0.6)
wth_patch = mpatches.Patch(fc=CLR_W, alpha=0.72, ec='none')
wth_dash  = Line2D([0],[0], ls=(0,(3,3)), lw=1.1, color=CLR_W, alpha=0.6)
hd = Line2D([], [], color='none')

ax.legend(
    handles=[(lfo_patch, lfo_dash), (wth_patch, wth_dash), hd],
    labels=[f'LFO (macro F1 = {ml:.3f})',
            f'Within-family (macro F1 = {mw:.3f})',
            'Δ = within − LFO'],
    handler_map={(tuple,): HandlerTuple(ndivide=None, pad=1)},
    frameon=False, fontsize=11.5,
    loc='lower center', handlelength=1.8, handletextpad=0.45,
    columnspacing=1.4, ncol=3, bbox_to_anchor=(0.44, -0.13)
)

plt.tight_layout(pad=0.5)
out = BASE / 'violin_paper'
fig.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(str(out) + '.pdf', bbox_inches='tight', facecolor='white')
print(f'Saved: {out}.pdf / .png')
