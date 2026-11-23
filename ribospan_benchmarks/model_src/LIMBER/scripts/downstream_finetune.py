#!/usr/bin/env python3
"""Unified downstream-task fine-tuning across all 7 RNA FM backbones (our
LIMBER-100M/300M/650M + baselines RiNALMo-giga, AIDO.RNA-650M, RNA-FM,
StructRFM), for two mRNA-region tasks the pretraining corpus (ncRNA-heavy
RNAcentral) barely covers:

  ires  - binary classification, IRES vs non-IRES, 174nt (DeepIRES benchmark)
  mrl   - regression, mean ribosomal loading, 25-100nt 5'UTRs (Sample et al.
          2019 / Karollus Random_Vary+Human_Vary benchmark)

Pooling is mean-pooling over the real-nucleotide positions (excludes
CLS/SEP/PAD) for every backbone -- uniform across models since none of them
were pretrained with a CLS-summary objective (structRFM's pooler is in fact
randomly initialized), so CLS-token pooling would not be a fair comparison.

Two-phase fine-tune: phase 1 head-only (backbone frozen), phase 2 full
unfreeze at a lower LR -- same recipe as nrc_finetune.py.

Usage:
  python downstream_finetune.py --model_type fm_hybrid_650m --task ires --out_dir ...
  python downstream_finetune.py --model_type structrfm_86m  --task mrl  --out_dir ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM
import ss_perseq_external as ext  # reuse baseline loaders + CKPT_DIRS/MODEL_NAMES
from _paths import PROJ

# ── Our own LIMBER checkpoints ──────────────────────────────────────────────
OUR_CKPT_DIRS = {
    "fm_hybrid_100m": str(PROJ / "checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch12"),
    "fm_hybrid_300m": str(PROJ / "checkpoints/fm_hybrid_300m/epoch_checkpoints/epoch12"),
    "fm_hybrid_650m": str(PROJ / "checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12"),
}
OUR_MODEL_NAMES = {
    "fm_hybrid_100m": "LIMBER-100M",
    "fm_hybrid_300m": "LIMBER-300M",
    "fm_hybrid_650m": "LIMBER-650M",
}
ALL_MODEL_TYPES = list(OUR_CKPT_DIRS) + list(ext.CKPT_DIRS)
ALL_MODEL_NAMES = {**OUR_MODEL_NAMES, **ext.MODEL_NAMES}

TOKENIZER_VOCAB = PROJECT_ROOT / "tokenizers" / "single" / "vocab.txt"
MAX_LEN = 1022

# Per-model max USABLE nucleotide content length (excludes CLS/SEP), so a
# batch never exceeds a backbone's positional-embedding table. RiNALMo-giga
# and RNA-FM report no max_position_embeddings (rotary/no hard limit
# observed) -> generous cap; AIDO.RNA and StructRFM have hard HF configs.
MODEL_MAX_CONTENT_LEN = {
    "fm_hybrid_100m": 1022,
    "fm_hybrid_300m": 1022,
    "fm_hybrid_650m": 1022,
    "rinalmo_giga":   2000,
    "aido_rna_650m":  1022,
    "rnafm_100m":     1022,
    "structrfm_86m":  512,
}


# ── Model loading ────────────────────────────────────────────────────────────

def _load_ours(ckpt_dir, device):
    from transformers import BertTokenizerFast
    cfg = json.load(open(Path(ckpt_dir) / "model_config.json"))
    keep = ("vocab_size", "hidden_size", "num_attention_heads", "intermediate_size",
            "max_position_embeddings", "type_vocab_size", "hidden_dropout_prob",
            "attention_probs_dropout_prob", "pad_token_id", "layer_pattern",
            "mamba_d_state", "mamba_d_conv", "mamba_expand", "tie_word_embeddings")
    model = HybridMambaForMaskedLM(HybridMambaConfig(**{k: cfg[k] for k in keep if k in cfg}))
    sd = torch.load(Path(ckpt_dir) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    tok = BertTokenizerFast(vocab_file=str(TOKENIZER_VOCAB), do_lower_case=False)
    layers = list(model.layers)
    hidden_size = model.config.hidden_size
    return model.to(device).eval(), tok, layers, hidden_size, True


def load_backbone(model_type, device):
    if model_type in OUR_CKPT_DIRS:
        return _load_ours(OUR_CKPT_DIRS[model_type], device)
    return ext.load_model(model_type, device)


# ── Tokenize + pool ──────────────────────────────────────────────────────────

def tokenize_ours(tok, seqs, device):
    """[CLS] + per-nt ids + [SEP], right-padded."""
    rows = []
    for s in seqs:
        ids = ([tok.cls_token_id]
               + [tok.convert_tokens_to_ids(c if c in "AUCGN" else "N") for c in s]
               + [tok.sep_token_id])
        rows.append(ids)
    maxL = max(len(r) for r in rows)
    ids_t = torch.zeros(len(rows), maxL, dtype=torch.long)
    am_t  = torch.zeros(len(rows), maxL, dtype=torch.long)
    for i, r in enumerate(rows):
        ids_t[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        am_t[i, :len(r)] = 1
    return ids_t.to(device), am_t.to(device)


def hidden_states(model_type, model, ids, am):
    """[B, L, d] last hidden states, unified across our HybridMamba model and
    baseline HF models."""
    if model_type in OUR_CKPT_DIRS:
        h = model.embeddings(input_ids=ids, token_type_ids=None)
        ext_mask = (1.0 - am[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for layer, is_m in zip(model.layers, model.is_mamba):
            h = layer(h) if is_m else layer(h, attention_mask=ext_mask)[0]
        return h
    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
    return out.last_hidden_state


def mean_pool(model_type, model, tok, seqs, device, has_special):
    """Mean-pool the real-nucleotide positions (CLS/SEP/PAD excluded) -> [B, d]."""
    cap = MODEL_MAX_CONTENT_LEN.get(model_type, 512)
    seqs = [s[:cap] for s in seqs]
    if model_type in OUR_CKPT_DIRS:
        ids, am = tokenize_ours(tok, seqs, device)
    else:
        enc = tok(list(seqs), return_tensors="pt", padding=True, add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        am  = enc.get("attention_mask", torch.ones_like(ids)).to(device)

    h = hidden_states(model_type, model, ids, am)   # [B, L, d]
    content = am.clone().float()
    if has_special:
        content[:, 0] = 0.0                          # drop CLS
        lens = am.sum(1) - 1                          # index of SEP (last real pos)
        content[torch.arange(content.size(0), device=device), lens] = 0.0
    denom = content.sum(1, keepdim=True).clamp(min=1.0)
    pooled = (h * content.unsqueeze(-1)).sum(1) / denom
    return pooled


class Head(nn.Module):
    def __init__(self, hidden_size, out_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size, hidden_size // 2)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_size // 2, out_dim)

    def forward(self, x):
        x = self.drop(self.norm(x))
        return self.fc2(self.act(self.fc1(x)))


# ── Data: IRES ───────────────────────────────────────────────────────────────

def load_ires_csv(path):
    import csv as _csv
    items = []
    with open(path) as f:
        r = _csv.DictReader(f)
        for row in r:
            items.append({"seq": row["seq"].upper().replace("T", "U"),
                          "label": int(row["label"])})
    return items


def load_ires_fasta(path):
    items, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(parts).upper().replace("T", "U")
                    lab = int(header.split("|")[1])
                    items.append({"seq": seq, "label": lab})
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        seq = "".join(parts).upper().replace("T", "U")
        lab = int(header.split("|")[1])
        items.append({"seq": seq, "label": lab})
    return items


def ires_datasets(root):
    root = Path(root)
    return {
        "train":       load_ires_csv(root / "dataset/train/train_set.csv"),
        "test_bal":    load_ires_csv(root / "dataset/test/balanced_test_set.csv"),
        "test_imbal":  load_ires_csv(root / "dataset/test/imbalanced_test_set.csv"),
        "test_main":   load_ires_fasta(root / "main_independent.fa"),
        "test_core":   load_ires_fasta(root / "core.fa"),
        "test_5utr":   load_ires_fasta(root / "5utr.fa"),
    }


# ── Data: MRL ────────────────────────────────────────────────────────────────

def mrl_datasets(path):
    import csv as _csv
    train, test_random, test_human = [], [], []
    with open(path) as f:
        r = _csv.DictReader(f)
        for row in r:
            item = {"seq": row["utr"].upper().replace("T", "U"),
                    "mrl": float(row["rl"]), "len": int(row["len"])}
            if row["set"] == "train" and row["library"] == "random":
                train.append(item)
            elif row["set"] == "test" and row["library"] == "random":
                test_random.append(item)
            elif row["set"] == "test" and row["library"] == "human":
                test_human.append(item)
    return train, test_random, test_human


# ── Metrics ──────────────────────────────────────────────────────────────────

def ires_metrics(y_true, y_prob):
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                  matthews_corrcoef, accuracy_score, balanced_accuracy_score,
                                  precision_score, recall_score, confusion_matrix)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    out = {"n": int(len(y_true)), "n_pos": int(y_true.sum())}
    multi_class = len(set(y_true.tolist())) > 1
    out["auroc"] = float(roc_auc_score(y_true, y_prob)) if multi_class else float("nan")
    out["auprc"] = float(average_precision_score(y_true, y_prob)) if multi_class else float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred.tolist())) > 1 else 0.0
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return out


def mrl_metrics(y_true, y_pred, lens=None):
    from scipy.stats import spearmanr, pearsonr
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    r2 = float(pearsonr(y_true, y_pred)[0] ** 2) if len(y_true) > 1 else float("nan")
    sp = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    out = {"n": int(len(y_true)), "r2": r2, "spearman": sp}
    if lens is not None:
        lens = np.asarray(lens)
        bins = [(25, 44), (45, 64), (65, 84), (85, 100)]
        by_bin = {}
        for lo, hi in bins:
            m = (lens >= lo) & (lens <= hi)
            if m.sum() > 1:
                by_bin[f"{lo}-{hi}"] = {
                    "n": int(m.sum()),
                    "r2": float(pearsonr(y_true[m], y_pred[m])[0] ** 2),
                    "spearman": float(spearmanr(y_true[m], y_pred[m])[0]),
                }
        out["by_length_bin"] = by_bin
    return out


# ── Training ─────────────────────────────────────────────────────────────────

def batches(items, bs, shuffle, rng):
    idx = np.arange(len(items))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i + bs]]


def run_task(model_type, model, tok, has_special, head, task, train, val_sets,
             device, out_dir, epochs1, epochs2, lr_head, lr_full, bs, seed):
    rng = np.random.default_rng(seed)
    is_ires = task == "ires"

    if is_ires:
        n_pos = sum(1 for it in train if it["label"] == 1)
        n_neg = len(train) - n_pos
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
        lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        lossfn = nn.HuberLoss(delta=1.0)

    def forward_batch(batch):
        seqs = [it["seq"] for it in batch]
        pooled = mean_pool(model_type, model, tok, seqs, device, has_special)
        logits = head(pooled).squeeze(-1)
        if is_ires:
            target = torch.tensor([it["label"] for it in batch], dtype=torch.float32, device=device)
        else:
            target = torch.tensor([it["mrl"] for it in batch], dtype=torch.float32, device=device)
        loss = lossfn(logits, target)
        return loss, logits, target

    @torch.no_grad()
    def evaluate(items, return_raw=False):
        model.eval(); head.eval()
        preds, targets, lens = [], [], []
        for batch in batches(items, bs, shuffle=False, rng=rng):
            _, logits, target = forward_batch(batch)
            if is_ires:
                preds.extend(torch.sigmoid(logits).cpu().tolist())
            else:
                preds.extend(logits.cpu().tolist())
            targets.extend(target.cpu().tolist())
            lens.extend([len(it["seq"]) for it in batch])
        metrics = ires_metrics(targets, preds) if is_ires else mrl_metrics(targets, preds, lens)
        if return_raw:
            return metrics, targets, preds
        return metrics

    def train_epochs(n_epochs, opt, phase):
        for ep in range(1, n_epochs + 1):
            model.train(any(p.requires_grad for p in model.parameters())); head.train()
            tot, nb = 0.0, 0
            for batch in batches(train, bs, shuffle=True, rng=rng):
                loss, _, _ = forward_batch(batch)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad], 1.0)
                opt.step()
                tot += loss.item(); nb += 1
            msg = f"  [{phase}] ep{ep:02d} train_loss={tot/max(nb,1):.4f}"
            primary = None
            for name, items in val_sets.items():
                m = evaluate(items)
                key = "auroc" if is_ires else "r2"
                msg += f"  {name}_{key}={m[key]:.4f}"
                if primary is None:
                    primary = m[key]
            print(msg, flush=True)
            yield primary

    for p_ in model.parameters():
        p_.requires_grad_(False)
    opt1 = torch.optim.AdamW(head.parameters(), lr=lr_head, weight_decay=0.01)
    best = -1e18
    best_head_sd = None
    for primary in train_epochs(epochs1, opt1, "P1"):
        if primary > best:
            best = primary
            best_head_sd = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_head_sd is not None:
        head.load_state_dict(best_head_sd)

    for p_ in model.parameters():
        p_.requires_grad_(True)
    opt2 = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": lr_full},
         {"params": head.parameters(), "lr": lr_full * 10}], weight_decay=0.01)
    best2 = -1e18
    best_head_sd2, best_model_sd2 = None, None
    for primary in train_epochs(epochs2, opt2, "P2"):
        if primary > best2:
            best2 = primary
            best_head_sd2 = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_model_sd2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_head_sd2 is not None:
        head.load_state_dict(best_head_sd2)
        model.load_state_dict(best_model_sd2)

    final, raw = {}, {}
    for name, items in val_sets.items():
        m, y_true, y_prob = evaluate(items, return_raw=True)
        final[name] = m
        raw[name] = {"y_true": y_true, "y_prob": y_prob}
    return final, raw


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=ALL_MODEL_TYPES)
    p.add_argument("--task", required=True, choices=["ires", "mrl"])
    p.add_argument("--ires_root", default=str(PROJ / "data/ires"))
    p.add_argument("--mrl_csv", default=str(PROJ / "data/mrl/varlen_mpra.csv"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs1", type=int, default=8)
    p.add_argument("--epochs2", type=int, default=12)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--lr_full", type=float, default=1e-5)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.task}_ft] model={args.model_type} device={device}", flush=True)

    model, tok, layers, hidden_size, has_special = load_backbone(args.model_type, device)
    print(f"[{args.task}_ft] {ALL_MODEL_NAMES[args.model_type]} hidden={hidden_size} "
          f"layers={len(layers)} has_special={has_special}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.task == "ires":
        head = Head(hidden_size, 1).to(device)
        ds = ires_datasets(args.ires_root)
        train_all = ds["train"]
        idx = rng.permutation(len(train_all))
        n_val = max(1, int(len(train_all) * args.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        train = [train_all[i] for i in tr_idx]
        val_sets = {
            "val":         [train_all[i] for i in val_idx],
            "test_bal":    ds["test_bal"],
            "test_imbal":  ds["test_imbal"],
            "test_main":   ds["test_main"],
            "test_core":   ds["test_core"],
            "test_5utr":   ds["test_5utr"],
        }
        print(f"[ires_ft] train={len(train)} val={len(val_sets['val'])} "
              f"test_bal={len(ds['test_bal'])} test_imbal={len(ds['test_imbal'])} "
              f"test_main={len(ds['test_main'])} test_core={len(ds['test_core'])} "
              f"test_5utr={len(ds['test_5utr'])}", flush=True)
    else:
        head = Head(hidden_size, 1).to(device)
        train_all, test_random, test_human = mrl_datasets(args.mrl_csv)
        idx = rng.permutation(len(train_all))
        n_val = max(1, int(len(train_all) * args.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        train = [train_all[i] for i in tr_idx]
        val_sets = {
            "val":         [train_all[i] for i in val_idx],
            "test_random": test_random,
            "test_human":  test_human,
        }
        print(f"[mrl_ft] train={len(train)} val={len(val_sets['val'])} "
              f"test_random={len(test_random)} test_human={len(test_human)}", flush=True)

    final, raw = run_task(args.model_type, model, tok, has_special, head, args.task,
                           train, val_sets, device, out_dir,
                           args.epochs1, args.epochs2, args.lr_head, args.lr_full,
                           args.batch_size, args.seed)

    result = {
        "model_type": args.model_type,
        "model_name": ALL_MODEL_NAMES[args.model_type],
        "task": args.task,
        "n_train": len(train),
        "results": final,
    }
    out_path = out_dir / "results.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\n[{args.task}_ft] Saved -> {out_path}", flush=True)
    for name, m in final.items():
        print(f"  {name}: {m}", flush=True)

    pred_path = out_dir / "predictions.json"
    json.dump(raw, open(pred_path, "w"))
    print(f"[{args.task}_ft] Raw predictions saved -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()
