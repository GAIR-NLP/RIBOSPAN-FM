#!/usr/bin/env python3
"""Hybrid-300M + RiNALMo SS pipeline with RiNALMo's GRADUAL-UNFREEZE fine-tuning.

Faithful to RiNALMo's training: epoch 0 trains only the head (LM frozen); then
every 3 epochs the next-deepest 3 LM layers are unfrozen (top-down) and added to
the optimizer as a new param group at base_lr/10. Base Adam lr 1e-5, LinearLR
1.0->0.1 over 7000 steps. Loss = BCE on the upper triangle (no pos_weight).
Threshold tuned on val (0.01..0.29, flexible F1); test uses RiNALMo's
canonical+greedy decoder and flexible(+/-1) F1. ArchiveII fam-fold split.
Only the backbone differs from RiNALMo (our hybrid-300M instead of RiNALMo LM).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc          # noqa: E402
from rinalmo_ss_lib import (SecStructPredictionHead, load_ss_dir, parse_ct,  # noqa: E402
                            prob_mat_to_sec_struct, ss_f1, ss_precision, ss_recall)

THRESHOLDS = [i / 100 for i in range(1, 30)]
LR_DECAY_STEPS = 7000        # RiNALMo LinearLR total_iters


def load_split(ct_dir, max_len):
    items, skipped = [], 0
    for ct in sorted(Path(ct_dir).glob("*.ct")):
        try:
            seq, pair = parse_ct(ct)
        except Exception:
            continue
        if 0 < len(seq) <= max_len:
            items.append({"id": ct.stem, "seq": seq, "pair": pair})
        else:
            skipped += 1
    return items, skipped


def backbone_hidden(model, ids, am):
    """Final hidden states of the hybrid encoder (skip MLM head). [B, T, d]."""
    h = model.embeddings(input_ids=ids, token_type_ids=None)
    ext = (1.0 - am[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
    for layer, is_m in zip(model.layers, model.is_mamba):
        h = layer(h) if is_m else layer(h, attention_mask=ext)[0]
    return h


def encode(tok, seq, device):
    ids = [tok.cls_token_id] + [tok.convert_tokens_to_ids(c) for c in seq] + [tok.sep_token_id]
    t = torch.tensor([ids], device=device)
    return t, torch.ones_like(t)


def logits_for(model, head, tok, seq, device):
    ids, am = encode(tok, seq, device)
    h = backbone_hidden(model, ids, am)[:, 1:1 + len(seq)]   # strip CLS/SEP -> [1,L,d]
    return head(h)[0]                                        # [L,L]


@torch.no_grad()
def eval_tune(model, head, tok, items, device):
    model.eval(); head.eval()
    per_t = {t: [] for t in THRESHOLDS}
    for it in items:
        probs = torch.sigmoid(logits_for(model, head, tok, it["seq"], device)).float().cpu().numpy()
        for t in THRESHOLDS:
            per_t[t].append(ss_f1(it["pair"], prob_mat_to_sec_struct(probs, it["seq"], threshold=t)))
    mean = {t: statistics.mean(v) if v else 0.0 for t, v in per_t.items()}
    bt = max(mean, key=mean.get)
    return bt, mean[bt]


@torch.no_grad()
def eval_test(model, head, tok, items, device, thr):
    model.eval(); head.eval()
    P, R, F, IDs = [], [], [], []
    for it in items:
        probs = torch.sigmoid(logits_for(model, head, tok, it["seq"], device)).float().cpu().numpy()
        ss = prob_mat_to_sec_struct(probs, it["seq"], threshold=thr)
        P.append(ss_precision(it["pair"], ss)); R.append(ss_recall(it["pair"], ss)); F.append(ss_f1(it["pair"], ss))
        IDs.append(it["id"])
    return P, R, F, IDs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="")
    p.add_argument("--model_dir", default="outputs/fm_hybrid_mamba_kmer1_300m")
    p.add_argument("--ct_root", default="data/contact_eval/raw/ct/fam-fold")
    p.add_argument("--out_dir", default="outputs/contact_pred/rinalmo_ss_ft")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=26)
    p.add_argument("--base_lr", type=float, default=5e-4)     # RiNALMo official SS-FT lr
    p.add_argument("--unfreeze_block", type=int, default=3)   # layers unfrozen per step
    p.add_argument("--unfreeze_every", type=int, default=3)   # epochs between unfreezes
    p.add_argument("--unfreeze_min_layer", type=int, default=0,
                   help="stop unfreezing at this layer idx (keep [0..min-1] frozen). giga keeps bottom 9.")
    p.add_argument("--tune_every", type=int, default=5)
    p.add_argument("--max_train", type=int, default=0, help="cap train items (0=all); for smoke tests")
    p.add_argument("--train_dir", default="", help="explicit train dir (.ct/.st/.bpseq); bypasses family folds")
    p.add_argument("--valid_dir", default="", help="explicit valid dir; if empty, hold out --val_frac of train")
    p.add_argument("--test_dir", default="", help="explicit test dir")
    p.add_argument("--test_dir2", default="", help="second test set, scored with the same tuned threshold")
    p.add_argument("--test_name", default="test")
    p.add_argument("--test_name2", default="test2")
    p.add_argument("--val_frac", type=float, default=0.10, help="held-out fraction of train when no valid_dir")
    p.add_argument("--tag", default="", help="output name + 'family' field (default: --family)")
    p.add_argument("--split_name", default="", help="label recorded in the result JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--within_family", action="store_true",
                   help="within-family split: use only this family's sequences (from test/ dir), "
                        "split 70/15/15 train/val/test instead of LFO.")
    p.add_argument("--within_train_frac", type=float, default=0.70)
    p.add_argument("--within_val_frac", type=float, default=0.15)
    args = p.parse_args()
    LABEL = args.tag or args.family

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = cc.load_model(args.model_dir, "hybrid", device)
    tok = cc.load_tokenizer("tokenizers/single")
    nlayers = len(model.layers)
    head = SecStructPredictionHead(model.config.hidden_size, num_blocks=2).to(device)

    root = Path(args.ct_root) / args.family
    if args.train_dir:
        train, sk_tr = load_ss_dir(args.train_dir, args.max_len, parse_ct)
        test, sk_te = load_ss_dir(args.test_dir, args.max_len, parse_ct)
        if args.valid_dir:
            val, sk_va = load_ss_dir(args.valid_dir, args.max_len, parse_ct)
        else:
            # Rivas has no official validation split: hold out a seeded fraction of train
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(len(train))
            n_va = max(1, int(len(train) * args.val_frac))
            val = [train[i] for i in idx[:n_va]]
            train = [train[i] for i in idx[n_va:]]
            sk_va = 0
        print(f"[{LABEL}] {args.split_name or 'explicit'} train/val/test = "
              f"{len(train)}/{len(val)}/{len(test)} (skipped>{args.max_len}nt: {sk_tr}/{sk_va}/{sk_te})  layers={nlayers}",
              flush=True)
    elif args.within_family:
        # Load all sequences of this family (stored in test/ for the LFO fold),
        # then do a random 70/15/15 split within-family.
        all_items, sk_all = load_split(root / "test", args.max_len)
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(all_items))
        n = len(idx)
        n_tr = max(1, int(n * args.within_train_frac))
        n_va = max(1, int(n * args.within_val_frac))
        train = [all_items[i] for i in idx[:n_tr]]
        val   = [all_items[i] for i in idx[n_tr:n_tr+n_va]]
        test  = [all_items[i] for i in idx[n_tr+n_va:]]
        sk_tr = sk_va = sk_te = sk_all
        print(f"[{LABEL}] WITHIN-FAMILY split {n} seqs → train={len(train)} val={len(val)} test={len(test)} "
              f"(skipped>{args.max_len}nt: {sk_all})  layers={nlayers}")
    else:
        train, sk_tr = load_split(root / "train", args.max_len)
        val, sk_va = load_split(root / "valid", args.max_len)
        test, sk_te = load_split(root / "test", args.max_len)
        print(f"[{LABEL}] LFO train/val/test = {len(train)}/{len(val)}/{len(test)} "
              f"(skipped>{args.max_len}nt: {sk_tr}/{sk_va}/{sk_te})  layers={nlayers}")
    if args.max_train:
        train = train[:args.max_train]

    # freeze backbone, head trainable
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in head.parameters():
        prm.requires_grad_(True)

    opt = torch.optim.Adam([{"params": list(head.parameters())}], lr=args.base_lr)
    base_lrs = [args.base_lr]

    # unfreeze schedule: epoch -> [layer indices], top-down 3 at a time,
    # stopping at unfreeze_min_layer (RiNALMo giga keeps the bottom 9 frozen).
    unfreeze_at = {}
    top = nlayers
    e = args.unfreeze_every
    while top > args.unfreeze_min_layer:
        lo = max(args.unfreeze_min_layer, top - args.unfreeze_block)
        unfreeze_at[e] = list(range(top - 1, lo - 1, -1))
        top = lo; e += args.unfreeze_every

    lossfn = nn.BCEWithLogitsLoss()
    gstep = 0
    unfrozen = set()                                  # layer indices currently trainable
    best_val_f1, best_thr, best_head, best_lm = -1.0, 0.5, None, None
    for ep in range(args.epochs):
        if ep in unfreeze_at:
            new_params = []
            for li in unfreeze_at[ep]:
                unfrozen.add(li)
                for prm in model.layers[li].parameters():
                    prm.requires_grad_(True); new_params.append(prm)
            opt.add_param_group({"params": new_params, "lr": args.base_lr / 10.0})
            base_lrs.append(args.base_lr / 10.0)
            print(f"  [{LABEL}] epoch {ep}: unfroze layers {unfreeze_at[ep]} (lr {args.base_lr/10:.1e})")

        # frozen backbone stays in eval mode (dropout off, as RiNALMo's freeze());
        # only the head + already-unfrozen layers run in train mode.
        model.eval(); head.train()
        for li in unfrozen:
            model.layers[li].train()
        order = np.random.permutation(len(train))
        tot = 0.0
        t0 = time.perf_counter()
        for k in order:
            it = train[k]; L = len(it["seq"])
            logits = logits_for(model, head, tok, it["seq"], device)
            ut = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            target = torch.from_numpy(it["pair"]).to(device)
            loss = lossfn(logits[ut], target[ut])
            opt.zero_grad(); loss.backward(); opt.step()
            gstep += 1
            factor = 1.0 - 0.9 * min(gstep / LR_DECAY_STEPS, 1.0)
            for gi, grp in enumerate(opt.param_groups):
                grp["lr"] = base_lrs[gi] * factor
            tot += loss.item()
        dt = time.perf_counter() - t0
        print(f"  [{LABEL}] epoch {ep} train {len(train)} items in {dt:.0f}s "
              f"({len(train)/max(dt,1e-9):.1f} it/s, {len(unfrozen)} layers unfrozen) loss={tot/len(train):.4f}", flush=True)
        if (ep + 1) % args.tune_every == 0 or ep == args.epochs - 1:
            thr, vf1 = eval_tune(model, head, tok, val, device)
            print(f"  [{LABEL}] epoch {ep+1} loss={tot/len(train):.4f} val_F1={vf1:.4f} thr={thr}", flush=True)
            if vf1 > best_val_f1:
                best_val_f1, best_thr = vf1, thr
                best_head = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                best_lm = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_head is not None:
        head.load_state_dict(best_head); model.load_state_dict(best_lm)
    P, R, F, IDs = eval_test(model, head, tok, test, device, best_thr)
    def _pct(lst, q): return float(np.percentile(lst, q)) if lst else 0.0
    split_mode = args.split_name or ("within_family" if args.within_family else "LFO")
    res = {
        "family": LABEL, "split": split_mode,
        "backbone": Path(args.model_dir).name,
        "pipeline": "RiNALMo+gradual-unfreeze-FT",
        "num_train": len(train), "num_val": len(val), "num_test": len(test),
        "tuned_threshold": best_thr, "best_val_F1": best_val_f1,
        "mean_F1":   statistics.mean(F)   if F else 0.0,
        "median_F1": statistics.median(F) if F else 0.0,
        "mean_precision":  statistics.mean(P) if P else 0.0,
        "mean_recall":     statistics.mean(R) if R else 0.0,
        "q25_F1": _pct(F, 25), "q75_F1": _pct(F, 75),
        "q10_F1": _pct(F, 10), "q90_F1": _pct(F, 90),
        "min_F1": min(F) if F else 0.0,  "max_F1": max(F) if F else 0.0,
        "per_seq_f1":       [round(f, 6) for f in F],
        "per_seq_precision": [round(v, 6) for v in P],
        "per_seq_recall":    [round(v, 6) for v in R],
        "per_seq_ids": IDs,
        "metric": "flexible(+/-1)_F1",
    }
    if args.test_dir2:
        items2, sk2 = load_ss_dir(args.test_dir2, args.max_len, parse_ct)
        P2, R2, F2, ID2 = eval_test(model, head, tok, items2, device, best_thr)
        res[args.test_name2] = {
            "num_test": len(items2), "skipped": sk2,
            "mean_F1": statistics.mean(F2) if F2 else 0.0,
            "median_F1": statistics.median(F2) if F2 else 0.0,
            "mean_precision": statistics.mean(P2) if P2 else 0.0,
            "mean_recall": statistics.mean(R2) if R2 else 0.0,
            "per_seq_f1": [round(v, 6) for v in F2],
            "per_seq_precision": [round(v, 6) for v in P2],
            "per_seq_recall": [round(v, 6) for v in R2],
            "per_seq_ids": ID2,
        }
        print(f"[{LABEL}] {args.test_name2} F1={res[args.test_name2]['mean_F1']:.4f} "
              f"median={res[args.test_name2]['median_F1']:.4f} n={len(items2)}", flush=True)
    res["test_name"] = args.test_name
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{LABEL}.json", "w"), indent=2)
    print(f"[{LABEL}] {split_mode} TEST F1={res['mean_F1']:.4f} median={res['median_F1']:.4f} "
          f"P={res['mean_precision']:.4f} R={res['mean_recall']:.4f} thr={best_thr} n={len(test)}", flush=True)


if __name__ == "__main__":
    main()
