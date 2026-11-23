#!/usr/bin/env python3
"""Aggregate the RiNALMo-pipeline ArchiveII leave-one-family-out results for the
hybrid-300M backbone, comparing FROZEN (head-only) vs GRADUAL-UNFREEZE FINE-TUNE.

Both use RiNALMo's exact head / BCE-on-upper-triangle loss / canonical+greedy
decoder / flexible(+/-1) F1 / val-threshold tuning / fam-fold splits. The only
difference between the two columns is whether the hybrid-300M backbone is frozen
or fine-tuned with RiNALMo's freeze-then-gradual-unfreeze schedule.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CP = Path("outputs/contact_pred")
FAMS = "5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA".split()


def load(sub, f):
    p = CP / sub / f"{f}.json"
    return json.load(open(p)) if p.exists() else None


def macro(rows, key):
    vals = [r[key] for r in rows.values() if r]
    return statistics.mean(vals) if vals else 0.0


frozen = {f: load("rinalmo_ss", f) for f in FAMS}
ft = {f: load("rinalmo_ss_ft", f) for f in FAMS}

print(f"\n{'family':12}{'frozen_F1':>11}{'FT_F1':>9}{'Δ':>8}{'FT_P':>8}{'FT_R':>8}{'thr':>6}{'n':>7}")
print("-" * 70)
for f in FAMS:
    fr, fe = frozen[f], ft[f]
    frf = fr["mean_F1"] if fr else float("nan")
    fef = fe["mean_F1"] if fe else float("nan")
    d = (fef - frf) if (fr and fe) else float("nan")
    p = fe["mean_precision"] if fe else float("nan")
    r = fe["mean_recall"] if fe else float("nan")
    thr = fe["tuned_threshold"] if fe else (fr["tuned_threshold"] if fr else "-")
    n = (fe or fr or {}).get("num_test", "-")
    print(f"{f:12}{frf:>11.3f}{fef:>9.3f}{d:>8.3f}{p:>8.3f}{r:>8.3f}{str(thr):>6}{str(n):>7}")

mfr = macro({f: frozen[f] for f in FAMS if frozen[f]}, "mean_F1")
mfe = macro({f: ft[f] for f in FAMS if ft[f]}, "mean_F1")
print("-" * 70)
print(f"{'MACRO':12}{mfr:>11.3f}{mfe:>9.3f}{mfe-mfr:>8.3f}")

out = {
    "backbone": "hybrid-300M",
    "pipeline": "RiNALMo SS (flexible F1, canonical+greedy decoder, val-threshold tuning)",
    "frozen_macro_F1": round(mfr, 4),
    "finetune_macro_F1": round(mfe, 4),
    "per_family": {f: {
        "frozen_F1": round(frozen[f]["mean_F1"], 4) if frozen[f] else None,
        "finetune_F1": round(ft[f]["mean_F1"], 4) if ft[f] else None,
        "finetune_precision": round(ft[f]["mean_precision"], 4) if ft[f] else None,
        "finetune_recall": round(ft[f]["mean_recall"], 4) if ft[f] else None,
        "finetune_threshold": ft[f]["tuned_threshold"] if ft[f] else None,
        "n_test": (ft[f] or frozen[f] or {}).get("num_test"),
    } for f in FAMS},
}
json.dump(out, open(CP / "compare_rinalmo_ss_ft_lfo.json", "w"), indent=2)

# grouped bar: frozen vs FT per family
fs = [f for f in FAMS if frozen[f] or ft[f]]
x = range(len(fs))
w = 0.4
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar([i - w / 2 for i in x], [frozen[f]["mean_F1"] if frozen[f] else 0 for f in fs],
       width=w, label=f"frozen (macro {mfr:.3f})", color="tab:gray")
ax.bar([i + w / 2 for i in x], [ft[f]["mean_F1"] if ft[f] else 0 for f in fs],
       width=w, label=f"gradual-unfreeze FT (macro {mfe:.3f})", color="tab:orange")
ax.set_xticks(list(x))
ax.set_xticklabels(fs, rotation=30)
ax.set_ylabel("flexible(+/-1) F1")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Hybrid-300M + RiNALMo SS pipeline — ArchiveII leave-one-family-out")
fig.tight_layout()
fig.savefig(CP / "compare_rinalmo_ss_ft_lfo.png", dpi=120)
print(f"\nwrote {CP}/compare_rinalmo_ss_ft_lfo.{{json,png}}")
