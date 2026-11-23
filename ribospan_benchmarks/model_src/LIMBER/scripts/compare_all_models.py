#!/usr/bin/env python3
"""Big cross-model comparison of every RNA foundation model trained here.

Crosses: tokenizer {kmer1, kmer6} x backbone/objective {BERT-MLM, hybrid-MLM,
decoder-causal}. All evaluated on the SAME validation base sequences, SAME
seed, unified to PER-BASE metrics so different vocabularies and objectives are
put on one ruler.

Per model: param count, eval loss (per-token), token accuracy, per-base
accuracy, per-base NLL, per-base perplexity, peak eval GPU memory, train
runtime, tokens/sec.

IMPORTANT caveat (printed in the output): MLM perplexity is a *pseudo*-PPL on
15% masked tokens with bidirectional context; causal (decoder) PPL is true
autoregressive PPL over every token with left-only context. Compare WITHIN an
objective for rigor; across objectives is indicative only.

Writes outputs/compare_all_models.json and outputs/compare_all_models.png.

Run:
    python scripts/compare_all_models.py
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
                          DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          set_seed)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402

OUT = PROJECT_ROOT / "outputs"

# tag, k, vocab_dir, model_dir, kind {bert|hybrid|gpt2}, objective {mlm|causal}, color
MODELS = [
    ("kmer1-BERT",    1, "tokenizers/single", "outputs/bert_mlm/fm_single",        "bert",   "mlm",    "tab:blue"),
    ("kmer1-hybrid",  1, "tokenizers/single", "outputs/fm_hybrid_mamba_kmer1",     "hybrid", "mlm",    "tab:cyan"),
    ("kmer1-decoder", 1, "tokenizers/single", "outputs/fm_decoder_kmer1",          "gpt2",   "causal", "tab:purple"),
    ("kmer6-BERT",    6, "tokenizers/kmer6",  "outputs/bert_mlm/fm_kmer6",         "bert",   "mlm",    "tab:green"),
    ("kmer6-hybrid",  6, "tokenizers/kmer6",  "outputs/fm_hybrid_mamba_kmer6",     "hybrid", "mlm",    "tab:olive"),
    ("kmer6-decoder", 6, "tokenizers/kmer6",  "outputs/fm_decoder_kmer6",          "gpt2",   "causal", "tab:red"),
]


def build_model(model_dir, kind):
    if kind == "bert":
        return BertForMaskedLM.from_pretrained(model_dir)
    if kind == "gpt2":
        return GPT2LMHeadModel.from_pretrained(model_dir)
    cfg = json.load(open(Path(model_dir) / "model_config.json"))
    keep = ("vocab_size", "hidden_size", "num_attention_heads", "intermediate_size",
            "max_position_embeddings", "type_vocab_size", "hidden_dropout_prob",
            "attention_probs_dropout_prob", "pad_token_id", "layer_pattern",
            "mamba_d_state", "mamba_d_conv", "mamba_expand", "tie_word_embeddings")
    model = HybridMambaForMaskedLM(HybridMambaConfig(**{k: cfg[k] for k in keep if k in cfg}))
    bin_path = Path(model_dir) / "pytorch_model.bin"
    sd = torch.load(bin_path, map_location="cpu") if bin_path.exists() else \
        __import__("safetensors.torch", fromlist=["load_file"]).load_file(
            str(Path(model_dir) / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    return model


def runtime_stats(model_dir, max_len):
    st = Path(model_dir) / "trainer_state.json"
    rt, sps = None, None
    if st.exists():
        for h in json.load(open(st)).get("log_history", []):
            if "train_runtime" in h:
                rt, sps = h["train_runtime"], h.get("train_samples_per_second")
    return {"train_runtime_s": rt,
            "tokens_per_sec": (sps * max_len) if sps else None}


def to_kmer_text(s, k):
    if k == 1:
        return " ".join(s)
    toks = []
    for i in range(0, len(s), k):
        t = s[i:i + k]
        toks.append(t if len(t) == k else t + "N" * (k - len(t)))
    return " ".join(toks)


@torch.no_grad()
def evaluate(model, tok, texts, k, objective, max_len, bs, device, seed):
    model = model.to(device).eval()
    id2tok = {v: kk for kk, v in tok.get_vocab().items()}
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=(objective == "mlm"),
        mlm_probability=0.15, pad_to_multiple_of=max_len)
    enc = tok(texts, truncation=True, max_length=max_len)
    feats = [{"input_ids": i, "attention_mask": a}
             for i, a in zip(enc["input_ids"], enc["attention_mask"])]
    loader = DataLoader(feats, batch_size=bs, collate_fn=collator)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    set_seed(seed)
    nll = 0.0
    tok_c = tok_n = base_c = base_n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=am).logits
        if objective == "causal":           # shift: pos t predicts t+1
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
        m = labels != -100
        nll += float(F.cross_entropy(logits[m], labels[m], reduction="sum"))
        preds = logits.argmax(-1)
        tok_c += int((preds[m] == labels[m]).sum())
        tok_n += int(m.sum())
        for pid, tid in zip(preds[m].tolist(), labels[m].tolist()):
            ptok, ttok = id2tok.get(pid, ""), id2tok.get(tid, "")
            if len(ttok) != k:          # special token ([SEP] etc): not a base k-mer
                continue
            pchars = ptok if len(ptok) == k else "?" * k
            for j in range(k):
                base_n += 1; base_c += (pchars[j] == ttok[j])
    L = nll / tok_n
    peak = (torch.cuda.max_memory_allocated() / 1e9) if device.startswith("cuda") else None
    return {"eval_loss_per_token": L, "token_acc": tok_c / tok_n,
            "per_base_acc": base_c / base_n, "per_base_nll": L / k,
            "per_base_ppl": math.exp(L / k), "masked_or_pred_tokens": tok_n,
            "evaluated_bases": base_n, "peak_eval_gpu_gb": peak}


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
    print(f"[cross] val seqs={len(base_seqs):,} device={device}\n")

    results = {}
    for tag, k, vocab_dir, mdir, kind, obj, color in MODELS:
        md = PROJECT_ROOT / mdir
        if not md.exists():
            print(f"[cross] SKIP {tag}: {md} missing"); continue
        tok = BertTokenizerFast(vocab_file=str(PROJECT_ROOT / vocab_dir / "vocab.txt"),
                                do_lower_case=False)
        max_len = 512 if k == 1 else 96
        texts = [to_kmer_text(s, k) for s in base_seqs]
        model = build_model(str(md), kind)
        n_params = sum(pp.numel() for pp in model.parameters())
        r = evaluate(model, tok, texts, k, obj, max_len, args.batch_size, device, args.seed)
        r.update(runtime_stats(str(md), max_len))
        r.update({"tokenizer": f"kmer{k}", "backbone": kind, "objective": obj,
                  "param_count": n_params, "color": color})
        results[tag] = r
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"  {tag:16} params={n_params:>10,}  per_base_acc={r['per_base_acc']*100:5.2f}%  "
              f"per_base_ppl={r['per_base_ppl']:.3f}  ({obj})")

    json.dump(results, open(OUT / "compare_all_models.json", "w"), indent=2)

    # ---- big figure: per-base accuracy + per-base PPL across all models ----
    tags = list(results.keys())
    colors = [results[t]["color"] for t in tags]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, title, better in (
        (a1, "per_base_acc", "Per-base accuracy (%)", "higher=better"),
        (a2, "per_base_ppl", "Per-base perplexity", "lower=better")):
        vals = [results[t][key] * (100 if key == "per_base_acc" else 1) for t in tags]
        bars = ax.bar(range(len(tags)), vals, color=colors)
        for r in bars:
            ax.annotate(f"{r.get_height():.3g}", (r.get_x()+r.get_width()/2, r.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{title}  ({better})")
        ax.grid(True, axis="y", alpha=0.3)
    a2.axhline(5.0, ls=":", color="gray")
    a2.annotate("random per-base = 5.0", (0, 5.05), color="gray", fontsize=8)
    fig.suptitle("RNA FM cross-model comparison — tokenizer × backbone/objective "
                 "(1 epoch, full 28M data, same val, unified per-base)\n"
                 "caveat: MLM pseudo-PPL (bidirectional, 15% masked) vs causal PPL "
                 "(left-only, every token) — compare within objective for rigor",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "compare_all_models.png", dpi=120)

    # ---- printed big table ----
    print("\n" + "=" * 104)
    hdr = f"{'model':16}{'tokenizer':10}{'objective':9}{'params':>11}{'tok_acc':>9}{'base_acc':>10}{'base_ppl':>10}{'gpu_gb':>8}"
    print(hdr); print("-" * 104)
    for t in tags:
        r = results[t]
        gb = f"{r['peak_eval_gpu_gb']:.2f}" if r['peak_eval_gpu_gb'] else "NA"
        print(f"{t:16}{r['tokenizer']:10}{r['objective']:9}{r['param_count']:>11,}"
              f"{r['token_acc']*100:>8.1f}%{r['per_base_acc']*100:>9.1f}%"
              f"{r['per_base_ppl']:>10.3f}{gb:>8}")
    print("=" * 104)
    print(f"wrote {OUT/'compare_all_models.json'} and {OUT/'compare_all_models.png'}")


if __name__ == "__main__":
    main()
