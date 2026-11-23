#!/usr/bin/env python3
"""Extract frozen hybrid-300M per-nucleotide embeddings for every unique
ArchiveII fam-fold sequence (<=512 nt), cached by sequence hash. Used by the
RiNALMo-pipeline secondary-structure runner.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc          # noqa: E402
from rinalmo_ss_lib import parse_ct  # noqa: E402

CACHE = PROJECT_ROOT / "outputs" / "contact_pred" / "rinalmo_ss" / "emb_300m"


def seq_hash(s):
    return hashlib.md5(s.encode()).hexdigest()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="outputs/fm_hybrid_mamba_kmer1_300m")
    p.add_argument("--ct_root", default="data/contact_eval/raw/ct/fam-fold")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default=None, help="override embedding cache dir")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    global CACHE
    if args.cache_dir:
        CACHE = Path(args.cache_dir)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = cc.load_model(args.model_dir, "hybrid", device)
    tok = cc.load_tokenizer("tokenizers/single")
    mm = cc.module_map(model, "hybrid")
    cap = {}
    mm["layers"][-1].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    # gather unique sequences (<=max_len) across all fam-fold CT files
    seqs = {}
    for ct in Path(args.ct_root).rglob("*.ct"):
        try:
            s, _ = parse_ct(ct)
        except Exception:
            continue
        if 0 < len(s) <= args.max_len:
            seqs.setdefault(seq_hash(s), s)
    todo = [(h, s) for h, s in seqs.items() if not (CACHE / f"{h}.pt").exists()]
    print(f"[extract] unique seqs<= {args.max_len}: {len(seqs)}  to extract: {len(todo)}")

    todo.sort(key=lambda x: len(x[1]))
    pad = tok.pad_token_id
    for s in range(0, len(todo), args.batch_size):
        chunk = todo[s:s + args.batch_size]
        maxL = max(len(x[1]) for x in chunk)
        ids = torch.full((len(chunk), maxL + 2), pad, dtype=torch.long)
        am = torch.zeros((len(chunk), maxL + 2), dtype=torch.long)
        for r, (_, seq) in enumerate(chunk):
            row = [tok.cls_token_id] + [tok.convert_tokens_to_ids(c) for c in seq] + [tok.sep_token_id]
            ids[r, :len(row)] = torch.tensor(row)
            am[r, :len(row)] = 1
        model(input_ids=ids.to(device), attention_mask=am.to(device))
        H = cap["h"].float().cpu()
        for r, (h, seq) in enumerate(chunk):
            torch.save(H[r, 1:1 + len(seq)].half(), CACHE / f"{h}.pt")
        if (s // args.batch_size) % 50 == 0:
            print(f"[extract] {s}/{len(todo)}")
    print(f"[extract] done -> {CACHE}")


if __name__ == "__main__":
    main()
