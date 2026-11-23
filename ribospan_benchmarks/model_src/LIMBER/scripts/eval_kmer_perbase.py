#!/usr/bin/env python3
"""Per-base accuracy AND per-base perplexity for one model (arbitrary k).

Same methodology as the single-vs-kmer3 headline numbers:
  - val base sequences, 15% MLM masking (HF collator, seed 42)
  - per-base accuracy: decode each predicted k-mer to k bases, compare to truth
  - per-base NLL: L_base = L_token / k ; PPL_base = exp(L_base)

single is k=1. Reusable for the full k-sweep.

Run:
    python scripts/eval_kmer_perbase.py --model_dir outputs/bert_mlm/kmer2_100k --k 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (BertForMaskedLM, BertTokenizerFast,
                          DataCollatorForLanguageModeling, set_seed)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLM_DIR = PROJECT_ROOT / "data" / "processed" / "mlm"


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


def to_kmer_text(s, k):
    if k == 1:
        return " ".join(s)
    toks = []
    for i in range(0, len(s), k):
        t = s[i:i + k]
        if len(t) < k:
            t = t + "N" * (k - len(t))
        toks.append(t)
    return " ".join(toks)


@torch.no_grad()
def evaluate(model_dir, k, val_file, cap_len, bs, device, seed):
    tok = BertTokenizerFast(vocab_file=str(Path(model_dir) / "vocab.txt"),
                            do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device).eval()
    id2tok = {v: kk for kk, v in tok.get_vocab().items()}
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True,
                                               mlm_probability=0.15)

    base_seqs = read_base_seqs(Path(val_file), cap_len, None)
    texts = [to_kmer_text(s, k) for s in base_seqs]
    enc = tok(texts, truncation=True, max_length=cap_len + 2)
    feats = [{"input_ids": i, "attention_mask": a}
             for i, a in zip(enc["input_ids"], enc["attention_mask"])]
    loader = DataLoader(feats, batch_size=bs, collate_fn=collator)

    set_seed(seed)
    tok_correct = tok_total = 0
    base_correct = base_total = 0
    total_nll = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        m = labels != -100
        total_nll += float(F.cross_entropy(logits[m], labels[m], reduction="sum"))
        preds = logits.argmax(-1)
        tok_correct += int((preds[m] == labels[m]).sum())
        tok_total += int(m.sum())
        for pid, tid in zip(preds[m].tolist(), labels[m].tolist()):
            ptok, ttok = id2tok.get(pid, ""), id2tok.get(tid, "")
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
        "L_token": L_token,
        "PPL_token": math.exp(L_token),
        "L_base": L_base,
        "PPL_base": math.exp(L_base),
        "masked_tokens": tok_total,
        "masked_bases": base_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--val_base_file", default=str(MLM_DIR / "val_single.txt"))
    p.add_argument("--cap_len", type=int, default=510)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r = evaluate(args.model_dir, args.k, args.val_base_file,
                 args.cap_len, args.batch_size, device, args.seed)
    print(json.dumps(r, indent=2))
    return r


if __name__ == "__main__":
    main()
