#!/usr/bin/env python3
"""Train the RiNALMo SecStructPredictionHead on frozen hybrid-300M embeddings
for ONE held-out ArchiveII family (RiNALMo fam-fold split), tune the decision
threshold on validation (candidates 0.01..0.29, maximize flexible F1), and
evaluate on the held-out family test set.

Pipeline (head, BCE-on-upper-triangle loss without pos_weight, canonical+greedy
decoder, flexible +/-1 F1, val-threshold tuning, fam-fold splits) is RiNALMo's.
The ONLY change is the backbone: RiNALMo LM -> our frozen hybrid-300M. Because
the backbone is frozen (head-only), the head is optimized with Adam lr 1e-3
(RiNALMo's 1e-5 is for full LM fine-tuning).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from rinalmo_ss_lib import (SecStructPredictionHead, parse_ct,  # noqa: E402
                            prob_mat_to_sec_struct, ss_f1, ss_precision, ss_recall)

CACHE = PROJECT_ROOT / "outputs" / "contact_pred" / "rinalmo_ss" / "emb_300m"
THRESHOLDS = [i / 100 for i in range(1, 30)]   # 0.01 .. 0.29 (RiNALMo)


def seq_hash(s):
    return hashlib.md5(s.encode()).hexdigest()


def load_split(ct_dir, max_len):
    items, skipped = [], 0
    for ct in sorted(Path(ct_dir).glob("*.ct")):
        try:
            seq, pair = parse_ct(ct)
        except Exception:
            continue
        if len(seq) > max_len or len(seq) == 0:
            skipped += 1
            continue
        emb_path = CACHE / f"{seq_hash(seq)}.pt"
        if not emb_path.exists():
            skipped += 1
            continue
        items.append({"id": ct.stem, "seq": seq, "pair": pair, "emb": emb_path})
    return items, skipped


def get_emb(item, device):
    return torch.load(item["emb"], weights_only=True).float().to(device)


@torch.no_grad()
def eval_threshold_tuning(head, items, device):
    """Return dict threshold -> mean flexible F1 over items (and best)."""
    head.eval()
    per_t = {t: [] for t in THRESHOLDS}
    for it in items:
        H = get_emb(it, device).unsqueeze(0)
        probs = torch.sigmoid(head(H)[0]).cpu().numpy()
        for t in THRESHOLDS:
            ss = prob_mat_to_sec_struct(probs, it["seq"], threshold=t)
            per_t[t].append(ss_f1(it["pair"], ss))
    mean = {t: statistics.mean(v) if v else 0.0 for t, v in per_t.items()}
    best_t = max(mean, key=mean.get)
    return best_t, mean[best_t]


@torch.no_grad()
def eval_test(head, items, device, threshold):
    head.eval()
    P, R, F = [], [], []
    for it in items:
        H = get_emb(it, device).unsqueeze(0)
        probs = torch.sigmoid(head(H)[0]).cpu().numpy()
        ss = prob_mat_to_sec_struct(probs, it["seq"], threshold=threshold)
        P.append(ss_precision(it["pair"], ss))
        R.append(ss_recall(it["pair"], ss))
        F.append(ss_f1(it["pair"], ss))
    return P, R, F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--ct_root", default="data/contact_eval/raw/ct/fam-fold")
    p.add_argument("--out_dir", default="outputs/contact_pred/rinalmo_ss")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--tune_every", type=int, default=5)
    p.add_argument("--cache_dir", default=None, help="embedding cache dir (per-epoch)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    global CACHE
    if args.cache_dir:
        CACHE = Path(args.cache_dir)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    root = Path(args.ct_root) / args.family
    train, sk_tr = load_split(root / "train", args.max_len)
    val, sk_va = load_split(root / "valid", args.max_len)
    test, sk_te = load_split(root / "test", args.max_len)
    print(f"[{args.family}] train/val/test = {len(train)}/{len(val)}/{len(test)} "
          f"(skipped >{args.max_len}nt or no-emb: {sk_tr}/{sk_va}/{sk_te})")

    head = SecStructPredictionHead(args.embed_dim, num_blocks=args.num_blocks).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1,
                                              total_iters=max(1, args.epochs * len(train)))
    lossfn = nn.BCEWithLogitsLoss()      # RiNALMo: no pos_weight

    best_val_f1, best_state, best_thr = -1.0, None, 0.5
    for ep in range(1, args.epochs + 1):
        head.train()
        order = np.random.permutation(len(train))
        tot = 0.0
        for k in order:
            it = train[k]
            H = get_emb(it, device).unsqueeze(0)
            logits = head(H)[0]
            L = len(it["seq"])
            ut = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            target = torch.from_numpy(it["pair"]).to(device)
            loss = lossfn(logits[ut], target[ut])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if ep % args.tune_every == 0 or ep == args.epochs:
            thr, vf1 = eval_threshold_tuning(head, val, device)
            print(f"  [{args.family}] epoch {ep:2d} loss={tot/len(train):.4f} "
                  f"val_F1={vf1:.4f} (thr={thr})")
            if vf1 > best_val_f1:
                best_val_f1, best_thr = vf1, thr
                best_state = {kk: v.cpu().clone() for kk, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    P, R, F = eval_test(head, test, device, best_thr)
    res = {
        "family": args.family, "backbone": "hybrid-300M", "pipeline": "RiNALMo",
        "num_train": len(train), "num_val": len(val), "num_test": len(test),
        "skipped_long_or_noemb": {"train": sk_tr, "val": sk_va, "test": sk_te},
        "tuned_threshold": best_thr, "best_val_F1": best_val_f1,
        "mean_F1": statistics.mean(F) if F else 0.0,
        "mean_precision": statistics.mean(P) if P else 0.0,
        "mean_recall": statistics.mean(R) if R else 0.0,
        "median_F1": statistics.median(F) if F else 0.0,
        "metric": "flexible(+/-1)_F1", "max_len": args.max_len,
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{args.family}.json", "w"), indent=2)
    print(f"[{args.family}] TEST F1={res['mean_F1']:.4f} P={res['mean_precision']:.4f} "
          f"R={res['mean_recall']:.4f} (thr={best_thr}, n={len(test)})")


if __name__ == "__main__":
    main()
