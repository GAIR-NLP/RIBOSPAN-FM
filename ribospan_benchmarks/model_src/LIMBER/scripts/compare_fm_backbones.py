#!/usr/bin/env python3
"""Compare vanilla-BERT vs Transformer+Mamba HYBRID backbones for the kmer1 AND
kmer6 tokenizers, on the SAME validation base sequences, SAME mask seed, SAME
15% MLM masking (=> same masked bases per tokenizer).

Per model: eval loss, token accuracy, per-base accuracy, per-base NLL, per-base
PPL, train runtime, tokens/sec, peak eval GPU memory, parameter count.
(kmer1: token acc == per-base acc.)

Writes outputs/fm_backbone_compare.json and outputs/fm_backbone_compare.png.

Run:
    python scripts/compare_fm_backbones.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (BertForMaskedLM, BertTokenizerFast,
                          DataCollatorForLanguageModeling, set_seed)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402

OUT = PROJECT_ROOT / "outputs"

# (tag, k, vocab_dir, bert_dir, hybrid_dir)
CONFIGS = [
    ("kmer1", 1, "tokenizers/single",
     "outputs/bert_mlm/fm_single", "outputs/fm_hybrid_mamba_kmer1"),
    ("kmer6", 6, "tokenizers/kmer6",
     "outputs/bert_mlm/fm_kmer6", "outputs/fm_hybrid_mamba_kmer6"),
]


def build_model(model_dir, kind):
    if kind == "bert":
        return BertForMaskedLM.from_pretrained(model_dir)
    cfg = json.load(open(Path(model_dir) / "model_config.json"))
    keep = ("vocab_size", "hidden_size", "num_attention_heads", "intermediate_size",
            "max_position_embeddings", "type_vocab_size", "hidden_dropout_prob",
            "attention_probs_dropout_prob", "pad_token_id", "layer_pattern",
            "mamba_d_state", "mamba_d_conv", "mamba_expand", "tie_word_embeddings")
    config = HybridMambaConfig(**{k: cfg[k] for k in keep if k in cfg})
    model = HybridMambaForMaskedLM(config)
    bin_path = Path(model_dir) / "pytorch_model.bin"
    if bin_path.exists():
        sd = torch.load(bin_path, map_location="cpu")
    else:
        from safetensors.torch import load_file
        sd = load_file(str(Path(model_dir) / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    return model


def runtime_stats(model_dir):
    st = Path(model_dir) / "trainer_state.json"
    out = {"train_runtime_s": None, "train_samples_per_second": None}
    if st.exists():
        for h in json.load(open(st)).get("log_history", []):
            if "train_runtime" in h:
                out["train_runtime_s"] = h["train_runtime"]
                out["train_samples_per_second"] = h.get("train_samples_per_second")
    return out


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
def evaluate(model, tok, texts, k, max_len, bs, device, seed):
    model = model.to(device).eval()
    id2tok = {v: kk for kk, v in tok.get_vocab().items()}
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=0.15, pad_to_multiple_of=max_len)
    enc = tok(texts, truncation=True, max_length=max_len)
    feats = [{"input_ids": i, "attention_mask": a}
             for i, a in zip(enc["input_ids"], enc["attention_mask"])]
    loader = DataLoader(feats, batch_size=bs, collate_fn=collator)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    set_seed(seed)
    nll = 0.0
    tok_correct = tok_total = 0
    base_correct = base_total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=am).logits
        m = labels != -100
        nll += float(F.cross_entropy(logits[m], labels[m], reduction="sum"))
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
    L = nll / tok_total
    peak = (torch.cuda.max_memory_allocated() / 1e9) if device.startswith("cuda") else None
    return {
        "eval_loss": L,
        "token_acc": tok_correct / tok_total,
        "per_base_acc": base_correct / base_total,
        "per_base_nll": L / k,
        "per_base_ppl": math.exp(L / k),
        "masked_tokens": tok_total,
        "masked_bases": base_total,
        "peak_eval_gpu_gb": peak,
    }


def read_base_seqs(path, cap, limit):
    seqs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                seqs.append("".join(line.split())[:cap])
            if limit and len(seqs) >= limit:
                break
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_base_file", default="/tmp/rna_fm_data/val_single.txt")
    p.add_argument("--max_val", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_seqs = read_base_seqs(Path(args.val_base_file), 510, args.max_val)
    print(f"[compare] val seqs={len(base_seqs):,} device={device}")

    results = {}
    for tag, k, vocab_dir, bert_dir, hybrid_dir in CONFIGS:
        tok = BertTokenizerFast(vocab_file=str(PROJECT_ROOT / vocab_dir / "vocab.txt"),
                                do_lower_case=False)
        max_len = 512 if k == 1 else 96
        texts = [to_kmer_text(s, k) for s in base_seqs]
        results[tag] = {}
        for name, mdir, kind in (("bert", bert_dir, "bert"),
                                 ("hybrid", hybrid_dir, "hybrid")):
            md = PROJECT_ROOT / mdir
            if not md.exists():
                print(f"[compare] SKIP {tag}/{name}: {md} missing")
                continue
            model = build_model(str(md), kind)
            n_params = sum(p.numel() for p in model.parameters())
            r = evaluate(model, tok, texts, k, max_len, args.batch_size, device, args.seed)
            r.update(runtime_stats(str(md)))
            r["param_count"] = n_params
            if r["train_samples_per_second"]:
                r["tokens_per_sec"] = r["train_samples_per_second"] * max_len
            results[tag][name] = r
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"[compare] {tag}/{name}: loss={r['eval_loss']:.4f} "
                  f"per_base_acc={r['per_base_acc']*100:.2f}% ppl_base={r['per_base_ppl']:.3f} "
                  f"params={n_params:,}")

    json.dump(results, open(OUT / "fm_backbone_compare.json", "w"), indent=2)

    # 2x2 plot: per-base accuracy and per-base PPL, BERT vs hybrid, for each tokenizer
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tags = [t for t, *_ in CONFIGS if t in results and len(results[t]) == 2]
    x = range(len(tags))
    w = 0.35
    for ax, key, title, better in (
        (axes[0], "per_base_acc", "Per-base accuracy (%)", "higher"),
        (axes[1], "per_base_ppl", "Per-base perplexity", "lower")):
        bert_v = [(results[t]["bert"][key] * (100 if key == "per_base_acc" else 1)) for t in tags]
        hyb_v = [(results[t]["hybrid"][key] * (100 if key == "per_base_acc" else 1)) for t in tags]
        b1 = ax.bar([i - w/2 for i in x], bert_v, w, label="BERT", color="tab:blue")
        b2 = ax.bar([i + w/2 for i in x], hyb_v, w, label="Hybrid (T+Mamba)", color="tab:orange")
        for bars in (b1, b2):
            for r in bars:
                ax.annotate(f"{r.get_height():.3g}", (r.get_x()+r.get_width()/2, r.get_height()),
                            ha="center", va="bottom", fontsize=9)
        ax.set_xticks(list(x)); ax.set_xticklabels(tags)
        ax.set_title(f"{title}  ({better}=better)")
        ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.suptitle("FM backbone comparison: vanilla BERT vs Transformer+Mamba hybrid "
                 "(same val, same mask, 1 epoch)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fm_backbone_compare.png", dpi=120)
    print(f"[compare] wrote {OUT/'fm_backbone_compare.json'} and .png")


if __name__ == "__main__":
    main()
