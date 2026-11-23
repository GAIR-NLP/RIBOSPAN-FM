#!/usr/bin/env python3
"""Zero-shot ncRNA classification on nRC dataset using pretrained embeddings.

Two evaluations:
  1. k-NN (k=1,5,10): train embeddings -> val labels (no fine-tuning)
  2. k-means (k=13) on val set only: cluster -> label matching via Hungarian
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from transformers import BertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402

MAX_LEN = 1022

CLASSES = [
    "5S_rRNA", "5_8S_rRNA", "CD-box", "HACA-box", "Intron_gpI", "Intron_gpII",
    "IRES", "leader", "miRNA", "ribozyme", "riboswitch", "scaRNA", "tRNA",
]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}


def parse_fasta(path):
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


def load_backbone(ckpt_dir, device):
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
    return model.to(device).eval()


@torch.no_grad()
def extract_embeddings(model, tok, items, device, batch_size=128):
    """Extract CLS token embeddings for all items. Returns (N, hidden_size) array."""
    embeddings, labels = [], []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        seqs = [it["seq"][:MAX_LEN] for it in batch]
        # tokenize with padding
        max_l = max(len(s) for s in seqs) + 2  # +CLS+SEP
        ids_batch, mask_batch = [], []
        for seq in seqs:
            ids = ([tok.cls_token_id]
                   + [tok.convert_tokens_to_ids(c if c in "AUCGN" else "N") for c in seq]
                   + [tok.sep_token_id])
            pad_len = max_l - len(ids)
            mask_batch.append([1] * len(ids) + [0] * pad_len)
            ids_batch.append(ids + [tok.pad_token_id] * pad_len)

        input_ids = torch.tensor(ids_batch, device=device)
        attention_mask = torch.tensor(mask_batch, device=device)

        h = model.embeddings(input_ids=input_ids, token_type_ids=None)
        ext = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for layer, is_m in zip(model.layers, model.is_mamba):
            h = layer(h) if is_m else layer(h, attention_mask=ext)[0]

        cls_h = h[:, 0].float().cpu().numpy()  # [B, hidden]
        embeddings.append(cls_h)
        labels.extend(LABEL2ID.get(it["label"], -1) for it in batch)

        if (i // batch_size) % 10 == 0:
            print(f"  [{i+len(batch)}/{len(items)}]", flush=True)

    return np.vstack(embeddings), np.array(labels)


def cluster_accuracy(y_true, y_pred, n_classes):
    """Hungarian matching: best assignment of cluster IDs to class labels."""
    cost = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cost[t, p] += 1
    row, col = linear_sum_assignment(-cost)
    mapping = {col[i]: row[i] for i in range(len(row))}
    mapped = np.array([mapping.get(p, -1) for p in y_pred])
    return accuracy_score(y_true, mapped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs/fm_hybrid_650m/epoch_checkpoints/epoch5")
    p.add_argument("--train_fasta", default="data/nrc/dataset_Rfam_6320_13classes.fasta")
    p.add_argument("--val_fasta", default="data/nrc/dataset_Rfam_validated_2600_13classes.fasta")
    p.add_argument("--tokenizer", default="tokenizers/single")
    p.add_argument("--out_dir", default="outputs/nrc_zeroshot")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[zero-shot] device={device}", flush=True)

    tok = BertTokenizerFast(vocab_file=str(root / args.tokenizer / "vocab.txt"),
                            do_lower_case=False)

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir

    print("[zero-shot] Loading model...", flush=True)
    model = load_backbone(ckpt_dir, device)

    train_items = parse_fasta(root / args.train_fasta)
    val_items = parse_fasta(root / args.val_fasta)

    print(f"[zero-shot] Extracting train embeddings ({len(train_items)})...", flush=True)
    train_emb, train_labels = extract_embeddings(model, tok, train_items, device, args.batch_size)

    print(f"[zero-shot] Extracting val embeddings ({len(val_items)})...", flush=True)
    val_emb, val_labels = extract_embeddings(model, tok, val_items, device, args.batch_size)

    results = {}

    # ---- 1. k-NN -------------------------------------------------------
    print("\n[zero-shot] === k-NN classification ===", flush=True)
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(train_emb, train_labels)
        preds = knn.predict(val_emb)
        acc = accuracy_score(val_labels, preds)
        print(f"  k={k:2d}  val_acc={acc:.4f}", flush=True)
        results[f"knn_k{k}"] = acc

    # ---- 2. k-means on val only ----------------------------------------
    print("\n[zero-shot] === k-means clustering (val only, k=13) ===", flush=True)
    km = KMeans(n_clusters=len(CLASSES), random_state=42, n_init=10)
    cluster_ids = km.fit_predict(val_emb)
    km_acc = cluster_accuracy(val_labels, cluster_ids, len(CLASSES))
    print(f"  k-means acc (Hungarian) = {km_acc:.4f}", flush=True)
    results["kmeans_val"] = km_acc

    # ---- 3. k-NN per-class breakdown ------------------------------------
    print("\n[zero-shot] === Per-class accuracy (1-NN) ===", flush=True)
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn1.fit(train_emb, train_labels)
    preds1 = knn1.predict(val_emb)
    per_class = {}
    for i, cls in enumerate(CLASSES):
        mask = val_labels == i
        n = mask.sum()
        acc = accuracy_score(val_labels[mask], preds1[mask]) if n > 0 else 0.0
        per_class[cls] = acc
        print(f"  {cls:<20s}  {acc:.3f}", flush=True)
    results["per_class_knn1"] = per_class

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[zero-shot] Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
