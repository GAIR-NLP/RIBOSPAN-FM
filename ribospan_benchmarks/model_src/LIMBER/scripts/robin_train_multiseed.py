#!/usr/bin/env python3
"""Is the per-molecule-split result (macro AUROC=0.641, AUPRC=0.032) actually
about the model, or just an artifact of which molecules happened to land in
train/val/test? Fixes the training recipe to the v3 winner (MLP head,
full-batch, natural pos_weight, 30 epochs + SWA-average last 5) and varies
ONLY the random seed controlling the compound-level split, across 10 seeds.
Reports the across-seed distribution of macro AUROC/AUPRC -- if it's as wide
as the 0.577-0.641 spread we saw across different TRAINING STRATEGIES on a
single fixed split, that spread was split-noise all along, not strategy.
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
_ap.add_argument("--out", default=str(ROOT / "multiseed_results.json"))
_ap.add_argument("--n_seeds", type=int, default=10)
_args = _ap.parse_args()
N_SEEDS = _args.n_seeds

# ------------------------------------------------------------------ load data (seed-independent)
rna_emb = np.load(_args.rna_emb)
rna_meta = json.load(open(ROOT / "rna_target_meta.json"))
rna_names = [m["target"] for m in rna_meta]

mol_emb = np.load(ROOT / "mol_embeddings.npy")
mol_names = json.load(open(ROOT / "mol_names.json"))

rows_all = list(csv.DictReader(open(ROOT / "SMM_Target_Hits.csv")))
row_by_name = {r["Name"]: r for r in rows_all}
rows = [row_by_name[n] for n in mol_names]

hit_cols = [f"{t}_hit" for t in rna_names]
Y = np.array([[float(r[c]) if r[c] not in ("", None) else 0.0 for c in hit_cols] for r in rows])
Y = (Y > 0).astype(np.float32)

N = len(mol_names)
n_rna = len(rna_names)
print(f"[multiseed] {N} molecules x {n_rna} RNA targets, {int(Y.sum())} positive pairs", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
rna_t = torch.tensor(rna_emb, dtype=torch.float32, device=device)
mol_t = torch.tensor(mol_emb, dtype=torch.float32, device=device)
Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
rna_t = (rna_t - rna_t.mean(0, keepdim=True)) / (rna_t.std(0, keepdim=True) + 1e-6)
mol_t = (mol_t - mol_t.mean(0, keepdim=True)) / (mol_t.std(0, keepdim=True) + 1e-6)
RNA_DIM, MOL_DIM = rna_t.shape[1], mol_t.shape[1]


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


def macro_metric(probs, y_idx, metric_fn):
    y_true = Y[y_idx]
    scores = []
    for i in range(n_rna):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(metric_fn(yt, probs[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def predict(model, mol_indices):
    model.eval()
    m_idx = torch.tensor(mol_indices, device=device)
    m_b = mol_t[m_idx]
    b = len(mol_indices)
    r_rep = rna_t.unsqueeze(0).expand(b, -1, -1).reshape(-1, RNA_DIM)
    m_rep = m_b.unsqueeze(1).expand(-1, n_rna, -1).reshape(-1, MOL_DIM)
    logits = model(r_rep, m_rep).reshape(b, n_rna)
    return torch.sigmoid(logits).cpu().numpy()


def run_one_seed(seed, epochs=30, batch_mols=256, avg_last_k=5):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    idx = rng.permutation(N)
    n_train = int(N * 0.7)
    n_val = int(N * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def iterate_full_epoch():
        order = train_idx.copy()
        rng.shuffle(order)
        for start in range(0, len(order), batch_mols):
            yield order[start:start + batch_mols]

    model = MLPHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_frac = Y[train_idx].mean()
    pos_weight_val = (1 - pos_frac) / max(pos_frac, 1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    tail_states = []
    for epoch in range(1, epochs + 1):
        model.train()
        for chunk in iterate_full_epoch():
            m_idx_t = torch.tensor(chunk, device=device)
            m_b = mol_t[m_idx_t]
            b = len(chunk)
            r_rep = rna_t.unsqueeze(0).expand(b, -1, -1).reshape(-1, RNA_DIM)
            m_rep = m_b.unsqueeze(1).expand(-1, n_rna, -1).reshape(-1, MOL_DIM)
            y_b = Y_t[m_idx_t].reshape(-1)
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

    test_probs = predict(model, test_idx)
    auroc = macro_metric(test_probs, test_idx, roc_auc_score)
    auprc = macro_metric(test_probs, test_idx, average_precision_score)
    return auroc, auprc


results = []
for seed in range(N_SEEDS):
    auroc, auprc = run_one_seed(seed)
    results.append({"seed": seed, "macro_auroc": auroc, "macro_auprc": auprc})
    print(f"[multiseed] seed {seed}  macro AUROC={auroc:.4f}  macro AUPRC={auprc:.4f}", flush=True)

aurocs = np.array([r["macro_auroc"] for r in results])
auprcs = np.array([r["macro_auprc"] for r in results])
print(f"\n[multiseed] AUROC: mean={aurocs.mean():.4f} std={aurocs.std():.4f} "
      f"min={aurocs.min():.4f} max={aurocs.max():.4f} range={aurocs.max()-aurocs.min():.4f}", flush=True)
print(f"[multiseed] AUPRC: mean={auprcs.mean():.4f} std={auprcs.std():.4f} "
      f"min={auprcs.min():.4f} max={auprcs.max():.4f} range={auprcs.max()-auprcs.min():.4f}", flush=True)

json.dump({"per_seed": results,
           "auroc_mean": float(aurocs.mean()), "auroc_std": float(aurocs.std()),
           "auroc_min": float(aurocs.min()), "auroc_max": float(aurocs.max()),
           "auprc_mean": float(auprcs.mean()), "auprc_std": float(auprcs.std()),
           "auprc_min": float(auprcs.min()), "auprc_max": float(auprcs.max())},
          open(_args.out, "w"), indent=2)
print(f"[multiseed] saved -> {_args.out}", flush=True)
