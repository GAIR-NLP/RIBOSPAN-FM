#!/usr/bin/env python3
"""Fine-tune Hybrid-650M on nRC 13-class ncRNA classification.

Two-phase training:
  Phase 1: backbone frozen, head only  (lr=5e-4, 10 epochs)
  Phase 2: backbone+head unfrozen      (lr=1e-5, 20 epochs)

Pooling: CLS token -> LayerNorm -> Dropout(0.1) -> Linear(1280, 13)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402

MAX_LEN = 1022   # usable residues; +CLS+SEP = 1024 tokens
MAX_LEN_BY_MODEL = {"structrfm_86m": 512}   # StructRFM's position embeddings stop at 512


CLASSES = [
    "5S_rRNA", "5_8S_rRNA", "CD-box", "HACA-box", "Intron_gpI", "Intron_gpII",
    "IRES", "leader", "miRNA", "ribozyme", "riboswitch", "scaRNA", "tRNA",
]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def parse_fasta(path: Path) -> list[dict]:
    items, header, seq_parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_parts).upper().replace("T", "U")
                    label = header.split()[1] if len(header.split()) > 1 else "UNK"
                    items.append({"seq": seq, "label": label})
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        seq = "".join(seq_parts).upper().replace("T", "U")
        label = header.split()[1] if len(header.split()) > 1 else "UNK"
        items.append({"seq": seq, "label": label})
    return items


class NRCDataset(Dataset):
    def __init__(self, items: list[dict], tok=None, encode_fn=None):
        self.items = items
        self.tok = tok
        self.encode_fn = encode_fn      # external backbones bring their own tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        seq = item["seq"][:MAX_LEN]
        if self.encode_fn is not None:
            label = LABEL2ID.get(item["label"], -1)
            return (torch.tensor(self.encode_fn(seq), dtype=torch.long),
                    torch.tensor(label, dtype=torch.long))
        ids = ([self.tok.cls_token_id]
               + [self.tok.convert_tokens_to_ids(c if c in "AUCGN" else "N")
                  for c in seq]
               + [self.tok.sep_token_id])
        label = LABEL2ID.get(item["label"], -1)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def make_collate(pad_id=0):
    def _collate(batch):
        ids_list, labels = zip(*batch)
        max_len = max(x.size(0) for x in ids_list)
        padded = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            padded[i, :ids.size(0)] = ids
            mask[i, :ids.size(0)] = 1
        return padded, mask, torch.stack(labels)
    return _collate


def collate(batch):
    ids_list, labels = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)
    padded = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :ids.size(0)] = ids
        mask[i, :ids.size(0)] = 1
    return padded, mask, torch.stack(labels)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_backbone(ckpt_dir: Path, device: torch.device) -> HybridMambaForMaskedLM:
    cfg = json.load(open(ckpt_dir / "model_config.json"))
    keep = ("vocab_size", "hidden_size", "num_attention_heads", "intermediate_size",
            "max_position_embeddings", "type_vocab_size", "hidden_dropout_prob",
            "attention_probs_dropout_prob", "pad_token_id", "layer_pattern",
            "mamba_d_state", "mamba_d_conv", "mamba_expand", "tie_word_embeddings")
    model = HybridMambaForMaskedLM(HybridMambaConfig(**{k: cfg[k] for k in keep if k in cfg}))
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.exists():
        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(str(ckpt_dir / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    return model.to(device)


def truncate_layers(model, n_layers):
    """Keep only the first n_layers encoder blocks (layer-importance ablation)."""
    total = len(model.layers)
    if not n_layers or n_layers >= total:
        return model
    model.layers = nn.ModuleList(list(model.layers)[:n_layers])
    model.is_mamba = list(model.is_mamba)[:n_layers]
    print(f"[nrc_ft] truncated encoder: {total} -> {len(model.layers)} layers "
          f"(pattern {''.join('M' if m else 'T' for m in model.is_mamba)})", flush=True)
    return model


def load_external(model_type, device):
    """Published baselines, each with its own tokenizer and special-token layout."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import ss_perseq_external as ext
    model, tok, _layers, hidden, has_special = ext.load_model(model_type, "cpu")
    if model_type == "rinalmo_giga":
        # from_pretrained renames "*.beta", leaving the trained SwiGLU betas at 1.0
        from safetensors import safe_open
        with safe_open(f"{ext.CKPT_DIRS[model_type]}/model.safetensors", "pt") as f:
            for i, layer in enumerate(model.layers):
                layer.ffn.beta.data.copy_(
                    f.get_tensor(f"model.layers.{i}.ffn.beta").to(layer.ffn.beta.dtype))
    model = model.to(device)
    model._ext_type = model_type
    model._has_special = has_special
    vocab = tok.get_vocab()
    u_to_t = "U" not in vocab and "T" in vocab          # RiNALMo uses a DNA alphabet
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def encode_fn(seq):
        s = seq.replace("U", "T") if u_to_t else seq
        return tok(s, add_special_tokens=True)["input_ids"]

    return model, hidden, encode_fn, pad_id


def backbone_cls(model, input_ids, attention_mask):
    """Return the pooled sequence representation: [B, hidden_size]."""
    if getattr(model, "_ext_type", None):
        h = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if model._has_special:
            return h[:, 0]                              # CLS
        m = attention_mask.unsqueeze(-1).to(h.dtype)    # StructRFM emits no CLS: mean-pool
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
    h = model.embeddings(input_ids=input_ids, token_type_ids=None)
    ext = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
    for layer, is_m in zip(model.layers, model.is_mamba):
        h = layer(h) if is_m else layer(h, attention_mask=ext)[0]
    return h[:, 0]  # CLS token


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, cls_h):
        return self.fc(self.drop(self.norm(cls_h)))


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def run_epoch(model, head, loader, optimizer, device, train: bool, accum: int = 1):
    """accum: gradient-accumulation steps, so micro-batch x accum is the effective batch."""
    model.train(train); head.train(train)
    total_loss = correct = total = 0
    step = -1
    ctx = torch.enable_grad() if train else torch.no_grad()

    def _step():
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if train:
        optimizer.zero_grad()
    with ctx:
        for step, (ids, mask, labels) in enumerate(loader):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            cls_h = backbone_cls(model, ids, mask)
            logits = head(cls_h)
            loss = nn.functional.cross_entropy(logits, labels)
            if train:
                (loss / accum).backward()
                if (step + 1) % accum == 0:
                    _step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    if train and step >= 0 and (step + 1) % accum != 0:
        _step()                                      # flush a partial final batch
    return total_loss / total, correct / total


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs/fm_hybrid_650m/epoch_checkpoints/epoch5")
    p.add_argument("--train_fasta", default="data/nrc/dataset_Rfam_6320_13classes.fasta")
    p.add_argument("--val_fasta", default="data/nrc/dataset_Rfam_validated_2600_13classes.fasta")
    p.add_argument("--tokenizer", default="tokenizers/single")
    p.add_argument("--out_dir", default="outputs/nrc_finetune")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--phase1_epochs", type=int, default=10)
    p.add_argument("--phase2_epochs", type=int, default=20)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--lr_full", type=float, default=1e-5)
    p.add_argument("--accum_steps", type=int, default=1,
                   help="gradient accumulation; batch_size x accum_steps = effective batch")
    p.add_argument("--n_layers", type=int, default=0,
                   help="keep only the first N encoder layers (0 = all)")
    p.add_argument("--model_type", default="",
                   choices=["", "rinalmo_giga", "aido_rna_650m", "rnafm_100m", "structrfm_86m"],
                   help="published baseline instead of a LIMBER checkpoint")
    args = p.parse_args()

    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nrc_ft] device={device}", flush=True)

    tok, encode_fn, pad_id, ext_model, ext_hidden = None, None, 0, None, None
    if args.model_type:
        global MAX_LEN
        MAX_LEN = min(MAX_LEN, MAX_LEN_BY_MODEL.get(args.model_type, MAX_LEN))
        ext_model, ext_hidden, encode_fn, pad_id = load_external(args.model_type, device)
        print(f"[nrc_ft] max_len={MAX_LEN}", flush=True)
        print(f"[nrc_ft] backbone={args.model_type} hidden={ext_hidden} pad_id={pad_id}", flush=True)
    else:
        tok = BertTokenizerFast(vocab_file=str(root / args.tokenizer / "vocab.txt"),
                                do_lower_case=False)

    train_items = parse_fasta(root / args.train_fasta)
    val_items = parse_fasta(root / args.val_fasta)
    print(f"[nrc_ft] train={len(train_items)}  val={len(val_items)}  classes={len(CLASSES)}")

    coll = make_collate(pad_id)
    train_loader = DataLoader(NRCDataset(train_items, tok, encode_fn), batch_size=args.batch_size,
                              shuffle=True, collate_fn=coll, num_workers=4, pin_memory=True)
    val_loader = DataLoader(NRCDataset(val_items, tok, encode_fn), batch_size=args.batch_size,
                            shuffle=False, collate_fn=coll, num_workers=4, pin_memory=True)

    if ext_model is not None:
        model, hidden_size = ext_model, ext_hidden
    else:
        ckpt_dir = Path(args.ckpt_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = root / ckpt_dir
        model = truncate_layers(load_backbone(ckpt_dir, device), args.n_layers)
        hidden_size = model.config.hidden_size
    head = ClassificationHead(hidden_size, len(CLASSES)).to(device)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: head only ------------------------------------------------
    print("\n[nrc_ft] === Phase 1: head only ===", flush=True)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    opt1 = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=0.01)
    best_val_acc = 0.0
    best_state = {"head": None, "model": None}

    for ep in range(1, args.phase1_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, head, train_loader, opt1, device, train=True, accum=args.accum_steps)
        va_loss, va_acc = run_epoch(model, head, val_loader, opt1, device, train=False)
        print(f"  P1 ep{ep:02d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}", flush=True)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state["head"] = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    # ---- Phase 2: full fine-tune -------------------------------------------
    print(f"\n[nrc_ft] === Phase 2: full fine-tune (best P1 val_acc={best_val_acc:.4f}) ===",
          flush=True)
    head.load_state_dict(best_state["head"])
    for p_ in model.parameters():
        p_.requires_grad_(True)

    opt2 = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.lr_full},
         {"params": head.parameters(), "lr": args.lr_full * 10}],
        weight_decay=0.01)

    best_val_acc2 = 0.0
    for ep in range(1, args.phase2_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, head, train_loader, opt2, device, train=True, accum=args.accum_steps)
        va_loss, va_acc = run_epoch(model, head, val_loader, opt2, device, train=False)
        print(f"  P2 ep{ep:02d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}", flush=True)
        if va_acc > best_val_acc2:
            best_val_acc2 = va_acc
            torch.save({
                "head": head.state_dict(),
                "model_ckpt": args.model_type or str(ckpt_dir),
                "classes": CLASSES,
                "val_acc": va_acc,
                "epoch": ep,
            }, out_dir / "best_model.pt")
            best_state["head"] = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_state["model"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[nrc_ft] Best val_acc = {best_val_acc2:.4f}", flush=True)

    # ---- Per-class accuracy on val -----------------------------------------
    head.load_state_dict(best_state["head"])
    model.load_state_dict(best_state["model"])
    model.eval(); head.eval()
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)
    all_prob, all_true = [], []
    with torch.no_grad():
        for ids, mask, labels in val_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = head(backbone_cls(model, ids, mask))
            all_prob.append(torch.softmax(logits.float(), dim=1).cpu())
            all_true.append(labels.cpu())
            preds = logits.argmax(1)
            for pred, true in zip(preds, labels):
                class_total[true.item()] += 1
                if pred == true:
                    class_correct[true.item()] += 1

    print("\n[nrc_ft] Per-class val accuracy:")
    for i, cls in enumerate(CLASSES):
        n = class_total[i]
        acc = class_correct[i] / n if n > 0 else 0.0
        print(f"  {cls:<20s}  {class_correct[i]:3d}/{n:3d}  {acc:.3f}")

    # ---- Full metric set on the 2,600-sequence eval split ------------------
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 matthews_corrcoef, roc_auc_score, average_precision_score)
    prob = torch.cat(all_prob).numpy()
    y_true = torch.cat(all_true).numpy()
    y_pred = prob.argmax(1)
    onehot = np.eye(len(CLASSES), dtype=np.int8)[y_true]
    metrics = {"n_eval": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
               "mcc": float(matthews_corrcoef(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f1)
        # one-vs-rest, so every class contributes a curve
        metrics[f"auroc_ovr_{avg}"] = float(roc_auc_score(onehot, prob, average=avg, multi_class="ovr"))
        metrics[f"auprc_ovr_{avg}"] = float(average_precision_score(onehot, prob, average=avg))
    per_class_full = {}
    p_c, r_c, f_c, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
    for i, cls in enumerate(CLASSES):
        per_class_full[cls] = {
            "n": int(sup[i]), "precision": float(p_c[i]), "recall": float(r_c[i]), "f1": float(f_c[i]),
            "auroc": float(roc_auc_score(onehot[:, i], prob[:, i])) if onehot[:, i].sum() else None,
            "auprc": float(average_precision_score(onehot[:, i], prob[:, i])) if onehot[:, i].sum() else None,
        }
    np.savez_compressed(out_dir / "eval_predictions.npz", prob=prob.astype(np.float32),
                        y_true=y_true.astype(np.int16), classes=np.array(CLASSES))
    print("\n[nrc_ft] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "n_eval"), flush=True)

    results = {
        "best_val_acc": best_val_acc2,
        "metrics": metrics,
        "per_class_full": per_class_full,
        "per_class": {CLASSES[i]: class_correct[i] / class_total[i]
                      for i in range(len(CLASSES)) if class_total[i] > 0},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[nrc_ft] Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
