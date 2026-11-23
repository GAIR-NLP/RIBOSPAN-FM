#!/usr/bin/env python3
"""Extract frozen-LM base-level embeddings for supervised contact prediction (Part 3).

kmer1: one token per base -> hidden states at base positions = [L, d].
kmer6: non-overlapping 6-mer tokens -> each token's hidden state is broadcast to
       its 6 bases; trailing pad-N bases are dropped so the output is [L, d].

Layer selected via --layer {final|embedding|<int>} (forward hook, weights frozen
and untouched). Saves outputs/contact_pred/embeddings/{model_name}/{dataset}/{id}.pt
(fp16 [L,d]) + a model-level metadata.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc  # noqa: E402

EMB_ROOT = PROJECT_ROOT / "outputs" / "contact_pred" / "embeddings"


def kmer6_tokens(seq, k=6):
    toks = []
    for i in range(0, len(seq), k):
        t = seq[i:i + k]
        toks.append(t if len(t) == k else t + "N" * (k - len(t)))
    return toks


def build_ids(tokenizer, seq, tokenizer_type):
    """Return (input_ids list incl CLS/SEP, n_content_tokens)."""
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    if tokenizer_type == "kmer1":
        content = [tokenizer.convert_tokens_to_ids(c) for c in seq]
    else:  # kmer6
        content = [tokenizer.convert_tokens_to_ids(t) for t in kmer6_tokens(seq)]
    return [cls] + content + [sep], len(content)


@torch.no_grad()
def extract_batch(model, capture, batch, device, tokenizer_type, pad_id):
    maxT = max(len(ids) for ids, _, _ in batch)
    inp = torch.full((len(batch), maxT), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), maxT), dtype=torch.long)
    for r, (ids, _, _) in enumerate(batch):
        inp[r, :len(ids)] = torch.tensor(ids)
        am[r, :len(ids)] = 1
    model(input_ids=inp.to(device), attention_mask=am.to(device))
    H = capture["h"].float().cpu()                         # [B, maxT, d]
    outs = []
    for r, (ids, ncontent, L) in enumerate(batch):
        content = H[r, 1:1 + ncontent]                     # [ntok, d]
        if tokenizer_type == "kmer1":
            base_emb = content[:L]
        else:
            idx = torch.arange(L) // 6                     # base i -> token i//6
            base_emb = content[idx]                        # [L, d]
        outs.append(base_emb)
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_type", choices=["bert", "hybrid"], required=True)
    p.add_argument("--tokenizer_type", choices=["kmer1", "kmer6"], required=True)
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--dataset_jsonl", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_name", default="archiveII")
    p.add_argument("--layer", default="final")
    p.add_argument("--max-len", type=int, default=512, dest="max_len")
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = cc.load_model(args.model_dir, args.model_type, device)
    tokenizer = cc.load_tokenizer(args.vocab_dir)
    mm = cc.module_map(model, args.model_type)
    if args.layer == "final":
        mod = mm["layers"][-1]
    elif args.layer == "embedding":
        mod = mm["embedding"]
    else:
        mod = mm["layers"][int(args.layer)]
    capture = {}
    mod.register_forward_hook(
        lambda m, i, o: capture.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    out_dir = EMB_ROOT / args.model_name / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(args.dataset_jsonl)]
    recs = [r for r in recs if r["length"] <= args.max_len]
    pad_id = tokenizer.pad_token_id

    # sort by length for efficient padded batching
    recs.sort(key=lambda r: r["length"])
    n_done = 0
    for s in range(0, len(recs), args.batch_size):
        chunk = recs[s:s + args.batch_size]
        batch = []
        for r in chunk:
            ids, ncontent = build_ids(tokenizer, r["sequence"], args.tokenizer_type)
            batch.append((ids, ncontent, r["length"]))
        embs = extract_batch(model, capture, batch, device, args.tokenizer_type, pad_id)
        for r, e in zip(chunk, embs):
            assert e.shape[0] == r["length"], (e.shape, r["length"])
            torch.save(e.half(), out_dir / f"{r['id']}.pt")
            n_done += 1
        if n_done % 200 < args.batch_size:
            print(f"[emb] {n_done}/{len(recs)}")

    meta = {"model_path": args.model_dir, "model_type": args.model_type,
            "tokenizer_type": args.tokenizer_type, "hidden_dim": model.config.hidden_size,
            "layer": args.layer, "num_sequences": n_done, "max_len": args.max_len,
            "dataset": args.dataset_name}
    json.dump(meta, open(EMB_ROOT / args.model_name / "metadata.json", "w"), indent=2)
    print(f"[emb] done {n_done} seqs -> {out_dir}  (hidden_dim={model.config.hidden_size})")


if __name__ == "__main__":
    main()
