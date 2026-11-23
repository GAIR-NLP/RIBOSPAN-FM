#!/usr/bin/env python3
import glob, json, re
from collections import defaultdict
import numpy as np
from _paths import REPO
R = str(REPO / "data/robin/multiseed_multi")
by = defaultdict(list)
for f in sorted(glob.glob(f"{R}/*.json")):
    name = re.match(r".*/(.+)\.json", f).group(1)
    d = json.load(open(f))
    by[name] = d
rows = []
for n, d in by.items():
    rows.append((n, len(d["per_seed"]), d["auroc_mean"], d["auroc_std"], d["auprc_mean"], d["auprc_std"]))
rows.sort(key=lambda r: -r[2])
hdr = "model_pool,n_seeds,macro_AUROC_mean,AUROC_std,macro_AUPRC_mean,AUPRC_std"
lines = [hdr] + [f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}" for r in rows]
open(f"{R}/summary.csv", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
