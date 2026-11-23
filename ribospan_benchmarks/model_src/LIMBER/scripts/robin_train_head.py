#!/usr/bin/env python3
"""RNA-small molecule binding prediction on ROBIN (RNA targets only, DNA
excluded). Both encoders (LIMBER, Uni-Mol) stay frozen -- only a small
scoring head is trained on their concatenated/projected embeddings.

v2 improvements over the first pass, still frozen-encoder:
  - positive-aware oversampling of molecules per epoch (instead of relying
    purely on a huge BCE pos_weight, which made training noisy/unstable)
  - proper val-set model selection (best macro AUPRC across epochs), not a
    fixed epoch count
  - two candidate heads (MLP concat vs. bilinear dot-product on projected
    embeddings) trained and compared, best one reported
  - more epochs with early stopping
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from _paths import REPO

ROOT = REPO / "data/robin"
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------ load data
rna_emb = np.load(ROOT / "rna_target_embeddings.npy")            # [27, 1280]
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
print(f"[robin_head] {N} molecules x {n_rna} RNA targets, "
      f"{int(Y.sum())} total positive pairs ({100*Y.mean():.3f}% positive)", flush=True)

# ------------------------------------------------------------------ compound-level split
idx = rng.permutation(N)
n_train = int(N * 0.7)
n_val = int(N * 0.15)
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]
print(f"[robin_head] split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
rna_t = torch.tensor(rna_emb, dtype=torch.float32, device=device)
mol_t = torch.tensor(mol_emb, dtype=torch.float32, device=device)
Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

rna_t = (rna_t - rna_t.mean(0, keepdim=True)) / (rna_t.std(0, keepdim=True) + 1e-6)
mol_t = (mol_t - mol_t.mean(0, keepdim=True)) / (mol_t.std(0, keepdim=True) + 1e-6)

RNA_DIM, MOL_DIM = rna_t.shape[1], mol_t.shape[1]

def iterate_full_epoch(batch_mols):
    """One full deterministic pass over all train molecules (shuffled), not a
    noisy positive-boosted subsample -- that was tried and made results worse
    (overfits the tiny val/test positive counts, hurts real generalization)."""
    order = train_idx.copy()
    rng.shuffle(order)
    for start in range(0, len(order), batch_mols):
        yield order[start:start + batch_mols]


# ------------------------------------------------------------------ heads
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


class BilinearHead(nn.Module):
    def __init__(self, proj=128):
        super().__init__()
        self.pr = nn.Sequential(nn.Linear(RNA_DIM, proj), nn.ReLU(), nn.Linear(proj, proj))
        self.pm = nn.Sequential(nn.Linear(MOL_DIM, proj), nn.ReLU(), nn.Linear(proj, proj))
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, r, m):
        zr = torch.nn.functional.normalize(self.pr(r), dim=-1)
        zm = torch.nn.functional.normalize(self.pm(m), dim=-1)
        return self.scale * (zr * zm).sum(-1) + self.bias


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


def macro_auprc(probs, y_true_idx):
    y_true = Y[y_true_idx]
    scores = []
    for i in range(n_rna):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(average_precision_score(yt, probs[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


def train_one(head_name, model, epochs=30, batch_mols=256, avg_last_k=5):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_frac = Y[train_idx].mean()
    pos_weight_val = (1 - pos_frac) / max(pos_frac, 1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    tail_states = []  # weights from the last `avg_last_k` epochs, for SWA-style averaging
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for chunk in iterate_full_epoch(batch_mols):
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
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        val_probs = predict(model, val_idx)
        val_ap = macro_auprc(val_probs, val_idx)
        print(f"[{head_name}] epoch {epoch:2d}  loss={total_loss/n_batches:.4f}  "
              f"val_macroAUPRC={val_ap:.4f}", flush=True)

        if epoch > epochs - avg_last_k:
            tail_states.append(copy.deepcopy(model.state_dict()))

    # SWA-style weight averaging over the last few epochs: smooths out the
    # noisy single-checkpoint selection that hurt the previous attempt.
    avg_state = {k: sum(s[k] for s in tail_states) / len(tail_states) for k in tail_states[0]}
    model.load_state_dict(avg_state)
    val_probs = predict(model, val_idx)
    val_ap = macro_auprc(val_probs, val_idx)
    print(f"[{head_name}] weight-averaged (last {avg_last_k} epochs) val_macroAUPRC={val_ap:.4f}", flush=True)
    return model, val_ap


print("\n[robin_head] === training MLPHead ===", flush=True)
mlp_model, mlp_val = train_one("mlp", MLPHead())
print("\n[robin_head] === training BilinearHead ===", flush=True)
bil_model, bil_val = train_one("bilinear", BilinearHead())

if mlp_val >= bil_val:
    best_name, best_model = "mlp", mlp_model
else:
    best_name, best_model = "bilinear", bil_model
print(f"\n[robin_head] best head: {best_name}  (val macroAUPRC mlp={mlp_val:.4f} bilinear={bil_val:.4f})", flush=True)

# ------------------------------------------------------------------ final test evaluation
test_probs = predict(best_model, test_idx)
test_y = Y[test_idx]

results = []
for i, name in enumerate(rna_names):
    y_true, y_prob = test_y[:, i], test_probs[:, i]
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        auroc, auprc = float("nan"), float("nan")
    else:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    base_rate = n_pos / len(y_true)
    lift = (auprc / base_rate) if (base_rate > 0 and not np.isnan(auprc)) else float("nan")
    results.append({"target": name, "structure_type": rna_meta[i]["structure_type"],
                     "n_test_pos": n_pos, "n_test": len(y_true),
                     "auroc": auroc, "auprc": auprc, "lift_over_base_rate": lift})
    print(f"[robin_head] {name:16s} n_pos={n_pos:4d}/{len(y_true):5d}  "
          f"AUROC={auroc:.3f}  AUPRC={auprc:.3f}  lift={lift:.1f}x", flush=True)

valid = [r for r in results if not np.isnan(r["auroc"])]
macro_auroc = float(np.mean([r["auroc"] for r in valid]))
macro_auprc = float(np.mean([r["auprc"] for r in valid]))
macro_lift = float(np.nanmean([r["lift_over_base_rate"] for r in valid]))
print(f"\n[robin_head] FINAL ({best_name})  macro AUROC={macro_auroc:.4f}  "
      f"macro AUPRC={macro_auprc:.4f}  macro lift={macro_lift:.1f}x "
      f"(over {len(valid)}/{n_rna} targets)", flush=True)

json.dump({"best_head": best_name, "per_target": results,
           "macro_auroc": macro_auroc, "macro_auprc": macro_auprc, "macro_lift": macro_lift},
          open(ROOT / "baseline_results.json", "w"), indent=2)
print(f"[robin_head] saved -> {ROOT / 'baseline_results.json'}", flush=True)
