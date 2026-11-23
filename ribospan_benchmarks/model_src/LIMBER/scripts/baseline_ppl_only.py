#!/usr/bin/env python3
"""Full-model (no ablation) masked-LM perplexity on the fixed 4096-seq val
set -- same methodology/data as layer_importance.py's baseline, but skips
the per-layer skip/prefix loops entirely.

Usage: python baseline_ppl_only.py --ckpt <dir> --out <dir> [--n_val 4096]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import torch

from hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM
from layer_importance import VAL_PATH, MAX_LEN, SEED, build_val_batches, run_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--n_val", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline_ppl] ckpt={args.ckpt} device={device}", flush=True)

    cd = json.load(open(Path(args.ckpt) / "model_config.json"))
    cfg = HybridMambaConfig(
        layer_pattern=cd["layer_pattern"], mamba_d_state=cd["mamba_d_state"],
        mamba_d_conv=cd["mamba_d_conv"], mamba_expand=cd["mamba_expand"],
        vocab_size=cd["vocab_size"], hidden_size=cd["hidden_size"],
        num_attention_heads=cd["num_attention_heads"],
        intermediate_size=cd["intermediate_size"],
        max_position_embeddings=cd["max_position_embeddings"],
        type_vocab_size=cd["type_vocab_size"], hidden_act=cd["hidden_act"],
        hidden_dropout_prob=cd["hidden_dropout_prob"],
        attention_probs_dropout_prob=cd["attention_probs_dropout_prob"],
        layer_norm_eps=cd["layer_norm_eps"],
        position_embedding_type=cd["position_embedding_type"],
    )
    model = HybridMambaForMaskedLM(cfg)
    state = torch.load(Path(args.ckpt) / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    n_layers = len(model.layers)
    print(f"[baseline_ppl] {n_layers} layers: {cfg.layer_pattern}", flush=True)

    batches = build_val_batches(VAL_PATH, MAX_LEN, args.n_val, SEED, args.batch)
    print(f"[baseline_ppl] {len(batches)} val batches ready", flush=True)

    ppl, seq_ppls, seq_lens = run_batches(model, batches, device, "full", k=None)
    print(f"[baseline_ppl] baseline PPL (all {n_layers} layers): {ppl:.4f}", flush=True)

    results = {
        "ckpt": args.ckpt,
        "layer_pattern": cfg.layer_pattern,
        "n_layers": n_layers,
        "baseline_ppl": ppl,
        "baseline_seq_ppls": seq_ppls,
        "baseline_seq_lens": seq_lens,
    }
    out_path = out_dir / "baseline_ppl.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[baseline_ppl] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
