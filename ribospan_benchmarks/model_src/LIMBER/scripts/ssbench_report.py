"""Build per-sequence CSVs + paired tests from the staged SS-benchmark JSONs."""
import json, glob, os, csv
import numpy as np
from scipy.stats import wilcoxon
from _paths import PROJ, REPO

S = str(PROJ / "outputs/ss_bench") + "/"
O = str(REPO / "_ssbench_out") + "/"
os.makedirs(O, exist_ok=True)
NAME = {"hybrid_100m": "LIMBER-nano (100M)", "hybrid_300m": "LIMBER-meso (300M)",
        "hybrid_650m": "LIMBER-macro (650M)", "rinalmo_giga": "RiNALMo-giga",
        "aido_rna_650m": "AIDO.RNA-650M", "rnafm_100m": "RNA-FM", "structrfm_86m": "StructRFM"}
ORDER = ["hybrid_650m", "hybrid_300m", "hybrid_100m", "rinalmo_giga", "aido_rna_650m", "structrfm_86m", "rnafm_100m"]

rows, byset = [], {}
for bench, main in (("bprna", "TS0"), ("rivas", "TestSetA")):
    for f in sorted(glob.glob(S + bench + "/*.json")):
        m = os.path.basename(f)[:-5]
        d = json.load(open(f))
        blocks = [(main, d["per_seq_ids"], d["per_seq_f1"], d["per_seq_precision"], d["per_seq_recall"])]
        if "TestSetB" in d:
            b = d["TestSetB"]
            blocks.append(("TestSetB", b["per_seq_ids"], b["per_seq_f1"], b["per_seq_precision"], b["per_seq_recall"]))
        for tset, ids, f1, pr, rc in blocks:
            for i, s in enumerate(ids):
                rows.append(dict(bench=bench, test_set=tset, model=m, model_name=NAME[m],
                                 seq_id=s, f1=f1[i], precision=pr[i], recall=rc[i]))
            byset.setdefault(tset, {})[m] = dict(zip(ids, f1))

with open(O + "per_sequence_f1.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["bench", "test_set", "model", "model_name", "seq_id", "f1", "precision", "recall"])
    w.writeheader(); w.writerows(rows)
print(f"per_sequence_f1.csv: {len(rows)} rows")

ref = "hybrid_650m"
out = []
for tset in ("TS0", "TestSetA", "TestSetB"):
    if tset not in byset or ref not in byset[tset]: continue
    for m in ORDER:
        if m == ref or m not in byset[tset]: continue
        ids = sorted(set(byset[tset][ref]) & set(byset[tset][m]))
        a = np.array([byset[tset][ref][i] for i in ids]); b = np.array([byset[tset][m][i] for i in ids])
        p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        out.append(dict(test_set=tset, n=len(ids), reference=NAME[ref], compared_with=NAME[m],
                        mean_F1_ref=round(a.mean(), 4), mean_F1_other=round(b.mean(), 4),
                        delta_F1=round(a.mean() - b.mean(), 4),
                        wins=int((a > b).sum()), losses=int((a < b).sum()), ties=int((a == b).sum()),
                        wilcoxon_p=f"{p:.3g}"))
with open(O + "paired_tests.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(f"paired_tests.csv: {len(out)} comparisons\n")
for r in out:
    print(f"  {r['test_set']:9s} vs {r['compared_with']:20s} n={r['n']:4d} dF1={r['delta_F1']:+.4f} "
          f"W/L={r['wins']}/{r['losses']} p={r['wilcoxon_p']}")
