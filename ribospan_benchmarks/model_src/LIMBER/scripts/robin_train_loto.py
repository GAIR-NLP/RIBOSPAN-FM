#!/usr/bin/env python3
"""Leave-one-RNA-target-out (LOTO) evaluation for the ROBIN binding predictor.

Unlike robin_train_head.py (which splits by MOLECULE and shares all 27 RNA
targets between train/test), this holds out one RNA target entirely: the
fusion head is trained on the other 26 targets x all molecules, then
evaluated on the held-out target x all molecules. Repeated once per target.
This measures generalization to an RNA the model never saw, not just to new
molecules for an RNA it already knows -- the gap the earlier per-molecule
split couldn't measure. Both encoders (LIMBER, Uni-Mol) stay frozen; only the
same small MLP fusion head architecture from the winning v3 recipe is used.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

import argparse
from _paths import REPO
ROOT = REPO / "data/robin"
_ap = argparse.ArgumentParser()
_ap.add_argument("--rna_emb", default=str(ROOT / "rna_target_embeddings.npy"))
_ap.add_argument("--out", default=str(ROOT / "loto_results.json"))
_ap.add_argument("--seed", type=int, default=0)
_args = _ap.parse_args()
SEED = _args.seed
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------ load data
rna_emb = np.load(_args.rna_emb)                                  # [27, d]
rna_meta = json.load(open(ROOT / "rna_target_meta.json"))
rna_names = [m["target"] for m in rna_meta]

mol_emb = np.load(ROOT / "mol_embeddings.npy")                    # [N, 512]
mol_names = json.load(open(ROOT / "mol_names.json"))

rows_all = list(csv.DictReader(open(ROOT / "SMM_Target_Hits.csv")))
row_by_name = {r["Name"]: r for r in rows_all}
rows = [row_by_name[n] for n in mol_names]

hit_cols = [f"{t}_hit" for t in rna_names]
Y = np.array([[float(r[c]) if r[c] not in ("", None) else 0.0 for c in hit_cols] for r in rows])
Y = (Y > 0).astype(np.float32)   # [N, 27]

N = len(mol_names)
n_rna = len(rna_names)
print(f"[loto] {N} molecules x {n_rna} RNA targets", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
rna_t_full = torch.tensor(rna_emb, dtype=torch.float32, device=device)
mol_t = torch.tensor(mol_emb, dtype=torch.float32, device=device)
Y_t_full = torch.tensor(Y, dtype=torch.float32, device=device)

# normalize using ALL 27 targets' stats (fixed, not refit per fold -- the
# RNA embedding space itself doesn't change based on which target is held out)
rna_t_full = (rna_t_full - rna_t_full.mean(0, keepdim=True)) / (rna_t_full.std(0, keepdim=True) + 1e-6)
mol_t = (mol_t - mol_t.mean(0, keepdim=True)) / (mol_t.std(0, keepdim=True) + 1e-6)

RNA_DIM, MOL_DIM = rna_t_full.shape[1], mol_t.shape[1]
ALL_MOL_IDX = np.arange(N)


class MLPHead(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(RNA_DIM + MOL_DIM, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, r, m):
        return self.net(torch.cat([r, m], dim=-1)).squeeze(-1)


def iterate_full_epoch(batch_mols):
    order = ALL_MOL_IDX.copy()
    rng.shuffle(order)
    for start in range(0, len(order), batch_mols):
        yield order[start:start + batch_mols]


@torch.no_grad()
def predict_one_target(model, rna_vec, mol_indices):
    model.eval()
    m_idx = torch.tensor(mol_indices, device=device)
    m_b = mol_t[m_idx]
    b = len(mol_indices)
    r_rep = rna_vec.unsqueeze(0).expand(b, -1)
    logits = model(r_rep, m_b)
    return torch.sigmoid(logits).cpu().numpy()


def train_fold(train_rna_idx, epochs=20, batch_mols=256, avg_last_k=5):
    model = MLPHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rna_train = rna_t_full[train_rna_idx]            # [26, 1280]
    Y_train = Y_t_full[:, train_rna_idx]              # [N, 26]
    n_train_rna = len(train_rna_idx)

    pos_frac = Y_train.mean().item()
    pos_weight_val = (1 - pos_frac) / max(pos_frac, 1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    tail_states = []
    for epoch in range(1, epochs + 1):
        model.train()
        for chunk in iterate_full_epoch(batch_mols):
            m_idx_t = torch.tensor(chunk, device=device)
            m_b = mol_t[m_idx_t]
            b = len(chunk)
            r_rep = rna_train.unsqueeze(0).expand(b, -1, -1).reshape(-1, RNA_DIM)
            m_rep = m_b.unsqueeze(1).expand(-1, n_train_rna, -1).reshape(-1, MOL_DIM)
            y_b = Y_train[m_idx_t].reshape(-1)

            logits = model(r_rep, m_rep)
            loss = loss_fn(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch > epochs - avg_last_k:
            tail_states.append(copy.deepcopy(model.state_dict()))

    avg_state = {k: sum(s[k] for s in tail_states) / len(tail_states) for k in tail_states[0]}
    model.load_state_dict(avg_state)
    return model


results = []
for held_out in range(n_rna):
    name = rna_names[held_out]
    train_rna_idx = [i for i in range(n_rna) if i != held_out]

    model = train_fold(train_rna_idx)
    probs = predict_one_target(model, rna_t_full[held_out], ALL_MOL_IDX)
    y_true = Y[:, held_out]
    n_pos = int(y_true.sum())

    if n_pos == 0 or n_pos == len(y_true):
        auroc, auprc = float("nan"), float("nan")
    else:
        auroc = roc_auc_score(y_true, probs)
        auprc = average_precision_score(y_true, probs)
    base_rate = n_pos / len(y_true)
    lift = (auprc / base_rate) if (base_rate > 0 and not np.isnan(auprc)) else float("nan")

    results.append({"held_out_target": name, "structure_type": rna_meta[held_out]["structure_type"],
                     "n_pos": n_pos, "n_total": len(y_true),
                     "auroc": auroc, "auprc": auprc, "lift_over_base_rate": lift})
    print(f"[loto] held out {name:16s} ({rna_meta[held_out]['structure_type']:20s}) "
          f"n_pos={n_pos:4d}/{len(y_true):5d}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  lift={lift:.1f}x",
          flush=True)

valid = [r for r in results if not np.isnan(r["auroc"])]
macro_auroc = float(np.mean([r["auroc"] for r in valid]))
macro_auprc = float(np.mean([r["auprc"] for r in valid]))
macro_lift = float(np.nanmean([r["lift_over_base_rate"] for r in valid]))
print(f"\n[loto] FINAL  macro AUROC={macro_auroc:.4f}  macro AUPRC={macro_auprc:.4f}  "
      f"macro lift={macro_lift:.1f}x  (over {len(valid)}/{n_rna} targets)", flush=True)

json.dump({"per_target": results, "macro_auroc": macro_auroc,
           "macro_auprc": macro_auprc, "macro_lift": macro_lift},
          open(_args.out, "w"), indent=2)
print(f"[loto] saved -> {_args.out}", flush=True)
