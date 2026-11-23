#!/usr/bin/env python3
"""
Layer importance analysis via two methods:
  1. Ablation: skip layer k, measure PPL with all other layers
  2. Prefix:   keep only layers 0..k-1, apply MLM head, measure PPL

Also stores per-sequence PPL for each (method, k) to allow downstream
analysis of which sequence types are most sensitive to each layer.

Usage: python layer_importance.py --ckpt <path> --out <dir> [--n_val 4096]
"""
import sys, os, json, math, random, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import torch
import torch.nn.functional as F
from rna_stream_dataset import MLMCollator, encode
from hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM
from _paths import PROJ

VAL_PATH = str(PROJ / "data/rnacentral/epoch_seqs_full/val.txt")
MAX_LEN  = 1022
SEED     = 42


def build_val_batches(val_path, max_len, n_max, seed, batch_size):
    rng  = random.Random(seed)
    coll = MLMCollator(seed=seed)
    feats = []
    with open(val_path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            ids, spec = encode(s, max_len, rng)
            feats.append({"input_ids": ids, "special_tokens_mask": spec})
            if len(feats) >= n_max:
                break
    return [coll(feats[i:i+batch_size]) for i in range(0, len(feats), batch_size)]


def run_batches(model, batches, device, mode, k):
    """
    Forward pass with mode='full'|'prefix'|'ablation' and layer index k.

    Returns:
        ppl          – aggregate per-token perplexity (float)
        seq_ppls     – list of per-sequence PPL values (one per sequence)
        seq_lens     – list of sequence lengths (number of masked tokens per seq)
    """
    total_loss = total_tok = 0
    seq_ppls = []
    seq_lens = []

    vocab_size = model.config.vocab_size

    for b in batches:
        b = {key: val.to(device) for key, val in b.items()}
        input_ids      = b["input_ids"]
        attention_mask = b["attention_mask"]
        labels         = b["labels"]

        if mode == "full":
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
        else:
            with torch.no_grad():
                hidden = model.embeddings(input_ids=input_ids)
                ext_mask = None
                if attention_mask is not None:
                    ext_mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype))
                    ext_mask = ext_mask * torch.finfo(hidden.dtype).min

                for i, (layer, is_m) in enumerate(zip(model.layers, model.is_mamba)):
                    if mode == "prefix"   and i >= k: break
                    if mode == "ablation" and i == k: continue
                    hidden = layer(hidden) if is_m else layer(hidden, attention_mask=ext_mask)[0]

                logits = model.cls(hidden)

        # per-token loss, shape (B, L)
        per_tok = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape)

        masked  = labels != -100          # (B, L)
        seq_sum = per_tok.sum(-1)         # (B,)
        seq_cnt = masked.sum(-1).float()  # (B,)

        for i in range(labels.shape[0]):
            n = int(seq_cnt[i].item())
            if n > 0:
                avg = seq_sum[i].item() / n
                seq_ppls.append(math.exp(min(avg, 20)))
                seq_lens.append(n)

        total_loss += seq_sum.sum().item()
        total_tok  += int(seq_cnt.sum().item())

    agg_ppl = math.exp(min(total_loss / max(total_tok, 1), 20))
    return agg_ppl, seq_ppls, seq_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--n_val",  type=int, default=4096)
    ap.add_argument("--batch",  type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[layer_importance] ckpt  : {args.ckpt}", flush=True)
    print(f"[layer_importance] device: {device}", flush=True)

    cd  = json.load(open(Path(args.ckpt) / "model_config.json"))
    cfg = HybridMambaConfig(
        layer_pattern                = cd["layer_pattern"],
        mamba_d_state                = cd["mamba_d_state"],
        mamba_d_conv                 = cd["mamba_d_conv"],
        mamba_expand                 = cd["mamba_expand"],
        vocab_size                   = cd["vocab_size"],
        hidden_size                  = cd["hidden_size"],
        num_attention_heads          = cd["num_attention_heads"],
        intermediate_size            = cd["intermediate_size"],
        max_position_embeddings      = cd["max_position_embeddings"],
        type_vocab_size              = cd["type_vocab_size"],
        hidden_act                   = cd["hidden_act"],
        hidden_dropout_prob          = cd["hidden_dropout_prob"],
        attention_probs_dropout_prob = cd["attention_probs_dropout_prob"],
        layer_norm_eps               = cd["layer_norm_eps"],
        position_embedding_type      = cd["position_embedding_type"],
    )
    model = HybridMambaForMaskedLM(cfg)
    state = torch.load(Path(args.ckpt) / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    n_layers      = len(model.layers)
    layer_pattern = cfg.layer_pattern
    print(f"[layer_importance] {n_layers} layers: {layer_pattern}", flush=True)

    print(f"[layer_importance] building val batches (n={args.n_val}) ...", flush=True)
    batches = build_val_batches(VAL_PATH, MAX_LEN, args.n_val, SEED, args.batch)
    print(f"[layer_importance] {len(batches)} batches ready", flush=True)

    results = {
        "ckpt":          args.ckpt,
        "layer_pattern": layer_pattern,
        "n_layers":      n_layers,
        "prefix":        [],
        "ablation":      [],
    }

    # baseline
    baseline, base_seq, base_lens = run_batches(model, batches, device, "full", k=None)
    results["baseline_ppl"]      = baseline
    results["baseline_seq_ppls"] = base_seq
    results["baseline_seq_lens"] = base_lens
    print(f"\n[layer_importance] baseline PPL (all {n_layers} layers): {baseline:.4f}", flush=True)

    # method 2: prefix
    print("\n--- Method 2: Prefix (keep first k layers) ---", flush=True)
    for k in range(1, n_layers + 1):
        ppl, seq_ppls, seq_lens = run_batches(model, batches, device, "prefix", k)
        lt = layer_pattern[k-1]
        results["prefix"].append({
            "k": k, "layer_type": lt, "ppl": ppl,
            "seq_ppls": seq_ppls, "seq_lens": seq_lens,
        })
        print(f"  prefix k={k:2d} (after {lt}): PPL={ppl:.4f}", flush=True)

    # method 1: ablation
    print("\n--- Method 1: Ablation (skip layer k) ---", flush=True)
    for k in range(n_layers):
        ppl, seq_ppls, seq_lens = run_batches(model, batches, device, "ablation", k)
        lt    = layer_pattern[k]
        delta = ppl - baseline
        results["ablation"].append({
            "k": k, "layer_type": lt, "ppl": ppl, "delta": delta,
            "seq_ppls": seq_ppls, "seq_lens": seq_lens,
        })
        print(f"  skip k={k:2d} ({lt}): PPL={ppl:.4f}  Δ={delta:+.4f}", flush=True)

    out_path = out_dir / "results.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[layer_importance] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
