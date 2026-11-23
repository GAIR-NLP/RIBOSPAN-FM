#!/usr/bin/env python3
"""Strictly mask-aligned per-base comparison across all k.

To hide the EXACT SAME bases for every tokenizer k in {1,2,3,4,5,6}, mask in
units of B = lcm(1..6) = 60 bases. Each sequence is split into consecutive
60-base blocks; a shared RNG selects ~15% of (full) blocks; every model then
masks the whole tokens covering each selected block (B is divisible by k, so
this hides exactly the 60 real bases of the block for every k).

=> identical hidden base set + identical count across all models. Reports
per-base accuracy and per-base NLL/perplexity on that common set.

Run:
    python scripts/eval_aligned_span.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLM_DIR = PROJECT_ROOT / "data" / "processed" / "mlm"
OUT_DIR = PROJECT_ROOT / "outputs" / "bert_mlm"

MODELS = [
    ("single", 1, OUT_DIR / "single_100k"),
    ("kmer2",  2, OUT_DIR / "kmer2_100k"),
    ("kmer3",  3, OUT_DIR / "kmer3_100k"),
    ("kmer4",  4, OUT_DIR / "kmer4_100k"),
    ("kmer5",  5, OUT_DIR / "kmer5_100k"),
    ("kmer6",  6, OUT_DIR / "kmer6_100k"),
]


def read_base_seqs(path, cap_len, limit):
    seqs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seqs.append("".join(line.split())[:cap_len])
            if limit and len(seqs) >= limit:
                break
    return seqs


def kmer_tokens(s, k):
    if k == 1:
        return list(s)
    toks = []
    for i in range(0, len(s), k):
        t = s[i:i + k]
        if len(t) < k:
            t = t + "N" * (k - len(t))
        toks.append(t)
    return toks


@torch.no_grad()
def eval_model(model_dir, k, seqs, masked_blocks, B, device, bs):
    tok = BertTokenizerFast(vocab_file=str(Path(model_dir) / "vocab.txt"),
                            do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device).eval()
    id2tok = {v: kk for kk, v in tok.get_vocab().items()}
    cls, sep, mask_id, pad_id = (tok.cls_token_id, tok.sep_token_id,
                                 tok.mask_token_id, tok.pad_token_id)

    examples = []   # (ids, [(pos, true_id), ...])
    for s, blocks in zip(seqs, masked_blocks):
        toks = kmer_tokens(s, k)
        ids = [cls] + [tok.convert_tokens_to_ids(t) for t in toks] + [sep]
        extra = []
        for b in blocks:
            base_start, base_end = b * B, b * B + B
            for ti in range(base_start // k, base_end // k):
                pos = 1 + ti
                extra.append((pos, ids[pos]))
                ids[pos] = mask_id
        if extra:
            examples.append((ids, extra))

    base_correct = base_total = 0
    tok_correct = tok_total = 0
    total_nll = 0.0
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        maxlen = max(len(e[0]) for e in chunk)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, (ids, _) in enumerate(chunk):
            input_ids[r, :len(ids)] = torch.tensor(ids)
            attn[r, :len(ids)] = 1
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=attn.to(device)).logits
        preds = logits.argmax(-1).cpu()
        logits = logits.cpu()
        for r, (ids, extra) in enumerate(chunk):
            for pos, true_id in extra:
                total_nll += float(F.cross_entropy(
                    logits[r, pos].unsqueeze(0),
                    torch.tensor([true_id]), reduction="sum"))
                tok_total += 1
                pid = int(preds[r, pos])
                tok_correct += (pid == true_id)
                ptok, ttok = id2tok.get(pid, ""), id2tok.get(true_id, "")
                if k == 1:
                    base_total += 1
                    base_correct += (ptok == ttok)
                else:
                    pchars = ptok if len(ptok) == k else "?" * k
                    for j in range(k):
                        base_total += 1
                        base_correct += (pchars[j] == ttok[j])

    L_token = total_nll / tok_total
    L_base = L_token / k
    return {
        "k": k,
        "token_acc": tok_correct / tok_total,
        "per_base_acc": base_correct / base_total,
        "L_base": L_base,
        "PPL_base": math.exp(L_base),
        "masked_bases": base_total,
        "masked_tokens": tok_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_base_file", default=str(MLM_DIR / "val_single.txt"))
    p.add_argument("--cap_len", type=int, default=510)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block_size", type=int, default=None,
                   help="Masked span length in bases (must be divisible by "
                        "every k; default = lcm(ks)).")
    p.add_argument("--out", default=str(OUT_DIR / "aligned_span_comparison.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ks = [m[1] for m in MODELS]
    B = args.block_size or math.lcm(*ks)
    for k in ks:
        assert B % k == 0, f"block_size {B} not divisible by k={k}"
    seqs = read_base_seqs(Path(args.val_base_file), args.cap_len, None)
    print(f"[aligned] val seqs: {len(seqs):,}  block B=lcm{tuple(ks)}={B}  device={device}")

    # Shared masked-block selection (identical for every model).
    rng = random.Random(args.seed)
    masked_blocks = []
    for s in seqs:
        nfull = len(s) // B
        masked_blocks.append([b for b in range(nfull) if rng.random() < args.mask_prob])
    n_masked_blocks = sum(len(b) for b in masked_blocks)
    print(f"[aligned] masked blocks: {n_masked_blocks:,} -> {n_masked_blocks*B:,} masked bases (same for all k)")

    results = []
    for name, k, mdir in MODELS:
        r = eval_model(str(mdir), k, seqs, masked_blocks, B, device, args.batch_size)
        r["model"] = name
        results.append(r)
        print(f"  {name:<8} k={k}  per_base_acc={r['per_base_acc']*100:5.2f}%  "
              f"PPL_base={r['PPL_base']:.3f}  masked_bases={r['masked_bases']:,}")

    out_path = args.out
    json.dump({"block_size": B, "mask_prob": args.mask_prob, "seed": args.seed,
               "results": results}, open(out_path, "w"), indent=2)
    print(f"\n[aligned] wrote {out_path}")


if __name__ == "__main__":
    main()
