#!/usr/bin/env python3
"""Extract CLS-token embeddings for ALL ArchiveII sequences (no per-family cap)
for the classification-figure PCA/LDA plot. Saves a dense [N, hidden] array +
matching family-label list.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc  # noqa: E402
from _paths import PROJ

CKPT = str(PROJ / "checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12")
VOCAB_DIR = str(PROJECT_ROOT / "tokenizers" / "single")
DATA = PROJECT_ROOT / "data" / "contact_eval" / "archiveII.jsonl"
OUT_EMB = PROJECT_ROOT / "outputs" / "family_embeddings_cls_full.npy"
OUT_LBL = PROJECT_ROOT / "outputs" / "family_embeddings_cls_full_labels.json"


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] loading model from {CKPT} on {device}", flush=True)
    model = cc.load_model(CKPT, "hybrid", device)
    tokenizer = cc.load_tokenizer(VOCAB_DIR)

    mm = cc.module_map(model, "hybrid")
    capture = {}
    mm["layers"][-1].register_forward_hook(
        lambda m, i, o: capture.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    recs = [json.loads(l) for l in open(DATA)]
    recs.sort(key=lambda r: r["length"])
    print(f"[extract] {len(recs)} total sequences", flush=True)

    embs = []
    labels = []
    for n, r in enumerate(recs, 1):
        ids, am = cc.encode_kmer1(tokenizer, r["sequence"], device)
        model(input_ids=ids, attention_mask=am)
        h = capture["h"][0, 0]  # CLS token, last layer
        embs.append(h.float().cpu().numpy())
        labels.append(r["family"])
        if n % 100 == 0 or n == len(recs):
            print(f"[extract] {n}/{len(recs)} done", flush=True)

    embs = np.stack(embs, axis=0)
    np.save(OUT_EMB, embs)
    json.dump(labels, open(OUT_LBL, "w"))
    print(f"[extract] saved {embs.shape} -> {OUT_EMB}", flush=True)
    from collections import Counter
    print(f"[extract] family counts: {Counter(labels)}", flush=True)


if __name__ == "__main__":
    main()
