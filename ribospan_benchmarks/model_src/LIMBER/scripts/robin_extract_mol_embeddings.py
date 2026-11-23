#!/usr/bin/env python3
"""Extract frozen Uni-Mol (unimolv1, 84m) embeddings for all ROBIN compounds."""
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from unimol_tools import UniMolRepr
from _paths import REPO

RDLogger.DisableLog("rdApp.*")

ROOT = REPO / "data/robin"
HITS_CSV = ROOT / "SMM_Target_Hits.csv"
OUT_EMB = ROOT / "mol_embeddings.npy"
OUT_NAMES = ROOT / "mol_names.json"
OUT_FAILED = ROOT / "mol_failed.json"

rows = list(csv.DictReader(open(HITS_CSV)))
names_all = [r["Name"] for r in rows]
smiles_all = [r["Smile"] for r in rows]
print(f"[robin_mol] {len(smiles_all)} compounds total", flush=True)

# pre-filter with RDKit: unimol_tools' own SMILES validator raises (not skips)
# on the first bad SMILES, so drop invalid ones ourselves first.
names, smiles, failed = [], [], []
for n, s in zip(names_all, smiles_all):
    mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    if mol is None:
        failed.append(n)
    else:
        names.append(n)
        smiles.append(s)
print(f"[robin_mol] {len(smiles)} valid, {len(failed)} dropped (invalid SMILES)", flush=True)
json.dump(failed, open(OUT_FAILED, "w"))

clf = UniMolRepr(data_type="molecule", model_name="unimolv1", model_size="84m",
                  use_cuda=True, batch_size=128)

reprs = clf.get_repr(smiles, return_tensor=False)
embs = np.array(reprs)
print(f"[robin_mol] embeddings shape {embs.shape}", flush=True)

np.save(OUT_EMB, embs)
json.dump(names, open(OUT_NAMES, "w"))
print(f"[robin_mol] saved -> {OUT_EMB}", flush=True)
