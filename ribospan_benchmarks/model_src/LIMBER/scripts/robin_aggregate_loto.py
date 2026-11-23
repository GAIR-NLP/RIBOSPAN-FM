#!/usr/bin/env python3
import glob, json, re
from collections import defaultdict
import numpy as np
from _paths import REPO
R = str(REPO / "data/robin/loto_multi")
by = defaultdict(list)
for f in sorted(glob.glob(f"{R}/*_seed*.json")):
    name = re.match(r".*/(.+)_seed\d+\.json", f).group(1)
    d = json.load(open(f)); by[name].append((d["macro_auroc"], d["macro_auprc"], d["macro_lift"]))
rows = []
for n, v in by.items():
    a = np.array(v)
    rows.append((n, len(v), a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(), a[:, 2].mean()))
rows.sort(key=lambda r: -r[2])
hdr = "model_pool,n_seeds,macro_AUROC,AUROC_std,macro_AUPRC,AUPRC_std,macro_lift"
lines = [hdr] + [f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.2f}" for r in rows]
open(f"{R}/summary.csv", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
