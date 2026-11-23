#!/usr/bin/env python3
"""SS fine-tune for external RNA foundation models: RiNALMo-giga, AIDO.RNA-650M, RNA-FM, StructRFM.

Same RiNALMo gradual-unfreeze FT protocol as rinalmo_ss_finetune.py; only the
backbone changes. Saves per_seq_f1 to JSON compatible with violin plot script.

Usage:
  python ss_perseq_external.py --model_type rinalmo_giga --family 5s --out_dir ...
  python ss_perseq_external.py --model_type aido_rna_650m --family 5s --out_dir ...
  python ss_perseq_external.py --model_type rnafm_100m --family 5s --out_dir ...
  python ss_perseq_external.py --model_type structrfm_86m --family 5s --out_dir ...
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
from rinalmo_ss_lib import (SecStructPredictionHead, load_ss_dir, parse_ct,  # noqa: E402
                            prob_mat_to_sec_struct, ss_f1, ss_precision, ss_recall)
from _paths import PROJ  # noqa: E402

THRESHOLDS = [i / 100 for i in range(1, 30)]
LR_DECAY_STEPS = 7000

# Checkpoint locations
CKPT_DIRS = {
    "rinalmo_giga":   str(PROJ / "checkpoints/rinalmo_giga"),
    "aido_rna_650m":  str(PROJ / "checkpoints/aido_rna_650m"),
    "rnafm_100m":     str(PROJ / "checkpoints/rnafm_100m"),
    "structrfm_86m":  str(PROJ / "checkpoints/structrfm_86m"),
}

MODEL_NAMES = {
    "rinalmo_giga":  "RiNALMo-giga",
    "aido_rna_650m": "AIDO.RNA-650M",
    "rnafm_100m":    "RNA-FM",
    "structrfm_86m": "StructRFM",
}


# ── Model loading ────────────────────────────────────────────────────────────

def _load_rinalmo(ckpt_dir, device):
    ckpt_dir = str(ckpt_dir)
    sys.path.insert(0, ckpt_dir)
    from transformers import AutoModel, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(ckpt_dir, trust_remote_code=True)
    layers = list(model.layers)          # 33 layers
    hidden_size = model.config.embed_dim
    return model.to(device).eval(), tok, layers, hidden_size, True


def _load_aido(ckpt_dir, device):
    """Direct loading because AutoModel hits a safetensors metadata bug."""
    ckpt_dir = Path(ckpt_dir)
    # Patch relative import once
    patched = ckpt_dir / "_modeling_aidorna_patched.py"
    if not patched.exists():
        src = (ckpt_dir / "modeling_aidorna.py").read_text()
        src = src.replace("from .configuration_aidorna", "from configuration_aidorna")
        patched.write_text(src)

    sys.path.insert(0, str(ckpt_dir))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_aidorna_model", patched)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AIDORNAModel = mod.AIDORNAModel

    from configuration_aidorna import AIDORNAConfig
    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    cfg_dict = json.load(open(ckpt_dir / "config.json"))
    exclude  = {"_name_or_path", "architectures", "transformers_version",
                "model_type", "auto_map", "torch_dtype"}
    cfg   = AIDORNAConfig(**{k: v for k, v in cfg_dict.items() if k not in exclude})
    model = AIDORNAModel(cfg)

    sd  = load_file(str(ckpt_dir / "model.safetensors"))
    sd2 = {(k[len("model."):] if k.startswith("model.") else k): v
           for k, v in sd.items()
           if not k.startswith("lm_head") and not k.startswith("cls")}
    model.load_state_dict(sd2, strict=False)

    tok    = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    layers = list(model.encoder.layer)   # 33 layers
    hidden_size = cfg.hidden_size
    return model.to(device).eval(), tok, layers, hidden_size, True


def _load_rnafm(ckpt_dir, device):
    ckpt_dir = str(ckpt_dir)
    sys.path.insert(0, ckpt_dir)
    from transformers import AutoModel, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(ckpt_dir, trust_remote_code=True)
    layers = list(model.layers)          # 12 layers
    hidden_size = model.config.embed_dim
    return model.to(device).eval(), tok, layers, hidden_size, True


def _load_structrfm(ckpt_dir, device):
    """Standard BERT, no CLS/SEP in tokenization, encoder.layer for layers."""
    from transformers import AutoModel, AutoTokenizer
    ckpt_dir = str(ckpt_dir)
    tok   = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModel.from_pretrained(ckpt_dir)
    layers = list(model.encoder.layer)  # 12 layers
    hidden_size = model.config.hidden_size
    return model.to(device).eval(), tok, layers, hidden_size, False


def load_model(model_type, device):
    ckpt_dir = CKPT_DIRS[model_type]
    loaders  = {
        "rinalmo_giga":  _load_rinalmo,
        "aido_rna_650m": _load_aido,
        "rnafm_100m":    _load_rnafm,
        "structrfm_86m": _load_structrfm,
    }
    return loaders[model_type](ckpt_dir, device)


# ── Encoding / hidden states ─────────────────────────────────────────────────

def encode(tok, seq, device):
    """Tokenise a single RNA sequence; add CLS + EOS/SEP."""
    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am  = enc.get("attention_mask", torch.ones_like(ids)).to(device)
    return ids, am


def backbone_hidden(model, tok, seq, device, has_special=True):
    """Final hidden states → [1, L, d]. Strip CLS/EOS only when has_special=True."""
    ids, am = encode(tok, seq, device)
    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
    h   = out.last_hidden_state
    if has_special:
        return h[:, 1:1 + len(seq)]      # strip CLS and SEP/EOS
    return h                              # StructRFM: no special tokens


def logits_for(model, head, tok, seq, device, has_special=True):
    h = backbone_hidden(model, tok, seq, device, has_special)   # [1, L, d]
    return head(h)[0]                                           # [L, L]


# ── Data loading ─────────────────────────────────────────────────────────────

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


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_tune(model, head, tok, items, device, has_special=True):
    model.eval(); head.eval()
    per_t = {t: [] for t in THRESHOLDS}
    for it in items:
        probs = torch.sigmoid(logits_for(model, head, tok, it["seq"], device, has_special)).float().cpu().numpy()
        for t in THRESHOLDS:
            per_t[t].append(ss_f1(it["pair"], prob_mat_to_sec_struct(probs, it["seq"], threshold=t)))
    mean = {t: statistics.mean(v) if v else 0.0 for t, v in per_t.items()}
    bt   = max(mean, key=mean.get)
    return bt, mean[bt]


@torch.no_grad()
def eval_test(model, head, tok, items, device, thr, has_special=True):
    model.eval(); head.eval()
    P, R, F, IDs = [], [], [], []
    for it in items:
        probs = torch.sigmoid(logits_for(model, head, tok, it["seq"], device, has_special)).float().cpu().numpy()
        ss    = prob_mat_to_sec_struct(probs, it["seq"], threshold=thr)
        P.append(ss_precision(it["pair"], ss))
        R.append(ss_recall(it["pair"], ss))
        F.append(ss_f1(it["pair"], ss))
        IDs.append(it["id"])
    return P, R, F, IDs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True,
                   choices=["rinalmo_giga", "aido_rna_650m", "rnafm_100m", "structrfm_86m"])
    p.add_argument("--family", default="")
    p.add_argument("--ct_root", default="data/contact_eval/raw/ct/fam-fold")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_len",  type=int, default=512)
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--base_lr",  type=float, default=5e-4)
    p.add_argument("--unfreeze_block",     type=int, default=3)
    p.add_argument("--unfreeze_every",     type=int, default=3)
    p.add_argument("--unfreeze_min_layer", type=int, default=3)
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
    p.add_argument("--within_family", action="store_true")
    p.add_argument("--within_train_frac", type=float, default=0.70)
    p.add_argument("--within_val_frac",   type=float, default=0.15)
    args = p.parse_args()
    LABEL = args.tag or args.family

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    model, tok, layers, hidden_size, has_special = load_model(args.model_type, device)
    nlayers = len(layers)
    head = SecStructPredictionHead(hidden_size, num_blocks=2).to(device)

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
        all_items, sk_all = load_split(root / "test", args.max_len)
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(all_items))
        n    = len(idx)
        n_tr = max(1, int(n * args.within_train_frac))
        n_va = max(1, int(n * args.within_val_frac))
        train = [all_items[i] for i in idx[:n_tr]]
        val   = [all_items[i] for i in idx[n_tr:n_tr + n_va]]
        test  = [all_items[i] for i in idx[n_tr + n_va:]]
        print(f"[{LABEL}] WITHIN-FAMILY {n} seqs → train={len(train)} val={len(val)} "
              f"test={len(test)} (skipped>{args.max_len}nt: {sk_all}) layers={nlayers}")
    else:
        train, sk_tr = load_split(root / "train", args.max_len)
        val,   sk_va = load_split(root / "valid", args.max_len)
        test,  sk_te = load_split(root / "test",  args.max_len)
        print(f"[{LABEL}] LFO train/val/test = {len(train)}/{len(val)}/{len(test)} "
              f"(skipped>{args.max_len}nt: {sk_tr}/{sk_va}/{sk_te}) layers={nlayers}")

    if args.max_train:
        train = train[:args.max_train]

    # Freeze backbone; head trainable
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in head.parameters():
        prm.requires_grad_(True)

    opt      = torch.optim.Adam([{"params": list(head.parameters())}], lr=args.base_lr)
    base_lrs = [args.base_lr]

    # Build unfreeze schedule (top-down, 3 layers at a time)
    unfreeze_at = {}
    top = nlayers
    e   = args.unfreeze_every
    while top > args.unfreeze_min_layer:
        lo = max(args.unfreeze_min_layer, top - args.unfreeze_block)
        unfreeze_at[e] = list(range(top - 1, lo - 1, -1))
        top = lo; e += args.unfreeze_every

    lossfn = nn.BCEWithLogitsLoss()
    gstep  = 0
    unfrozen = set()
    best_val_f1, best_thr, best_head_sd, best_model_sd = -1.0, 0.5, None, None

    for ep in range(args.epochs):
        if ep in unfreeze_at:
            new_params = []
            for li in unfreeze_at[ep]:
                unfrozen.add(li)
                for prm in layers[li].parameters():
                    prm.requires_grad_(True); new_params.append(prm)
            opt.add_param_group({"params": new_params, "lr": args.base_lr / 10.0})
            base_lrs.append(args.base_lr / 10.0)
            print(f"  [{LABEL}] ep {ep}: unfroze layers {unfreeze_at[ep]} (lr {args.base_lr/10:.1e})")

        model.eval(); head.train()
        for li in unfrozen:
            layers[li].train()

        order = np.random.permutation(len(train))
        tot   = 0.0
        t0    = time.perf_counter()
        for k in order:
            it = train[k]; L = len(it["seq"])
            logits = logits_for(model, head, tok, it["seq"], device, has_special)
            ut     = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            target = torch.from_numpy(it["pair"]).to(device)
            loss   = lossfn(logits[ut], target[ut])
            opt.zero_grad(); loss.backward(); opt.step()
            gstep += 1
            factor = 1.0 - 0.9 * min(gstep / LR_DECAY_STEPS, 1.0)
            for gi, grp in enumerate(opt.param_groups):
                grp["lr"] = base_lrs[gi] * factor
            tot += loss.item()
        dt = time.perf_counter() - t0
        print(f"  [{LABEL}] ep {ep} {len(train)} items in {dt:.0f}s "
              f"({len(train)/max(dt,1e-9):.1f} it/s, {len(unfrozen)} unfrozen) loss={tot/len(train):.4f}", flush=True)

        if (ep + 1) % args.tune_every == 0 or ep == args.epochs - 1:
            thr, vf1 = eval_tune(model, head, tok, val, device, has_special)
            print(f"  [{LABEL}] ep {ep+1} val_F1={vf1:.4f} thr={thr}", flush=True)
            if vf1 > best_val_f1:
                best_val_f1, best_thr = vf1, thr
                best_head_sd  = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                best_model_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_head_sd is not None:
        head.load_state_dict(best_head_sd)
        model.load_state_dict(best_model_sd)

    P, R, F, IDs = eval_test(model, head, tok, test, device, best_thr, has_special)

    def _pct(lst, q): return float(np.percentile(lst, q)) if lst else 0.0
    split_mode = args.split_name or ("within_family" if args.within_family else "LFO")
    res = {
        "family":   args.family,
        "split":    split_mode,
        "backbone": MODEL_NAMES[args.model_type],
        "pipeline": "RiNALMo+gradual-unfreeze-FT",
        "num_train": len(train), "num_val": len(val), "num_test": len(test),
        "tuned_threshold": best_thr, "best_val_F1": best_val_f1,
        "mean_F1":         statistics.mean(F)   if F else 0.0,
        "median_F1":       statistics.median(F) if F else 0.0,
        "mean_precision":  statistics.mean(P)   if P else 0.0,
        "mean_recall":     statistics.mean(R)   if R else 0.0,
        "q25_F1": _pct(F, 25), "q75_F1": _pct(F, 75),
        "q10_F1": _pct(F, 10), "q90_F1": _pct(F, 90),
        "min_F1": min(F) if F else 0.0, "max_F1": max(F) if F else 0.0,
        "per_seq_f1":        [round(f, 6) for f in F],
        "per_seq_precision": [round(v, 6) for v in P],
        "per_seq_recall":    [round(v, 6) for v in R],
        "per_seq_ids": IDs,
        "metric": "flexible(+/-1)_F1",
    }
    if args.test_dir2:
        items2, sk2 = load_ss_dir(args.test_dir2, args.max_len, parse_ct)
        P2, R2, F2, ID2 = eval_test(model, head, tok, items2, device, best_thr, has_special)
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
    print(f"[{LABEL}] {split_mode} TEST F1={res['mean_F1']:.4f} "
          f"median={res['median_F1']:.4f} P={res['mean_precision']:.4f} "
          f"R={res['mean_recall']:.4f} thr={best_thr} n={len(test)}", flush=True)


if __name__ == "__main__":
    main()
