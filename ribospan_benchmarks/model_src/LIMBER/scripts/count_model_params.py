#!/usr/bin/env python3
"""Compare parameter counts of the vanilla BERT kmer1 backbone vs the
Transformer+Mamba hybrid backbone, with an embedding / encoder / MLM-head
breakdown. Goal: hybrid total within +/-10% of the BERT baseline.

Writes outputs/model_param_compare.json.

Run:
    python scripts/count_model_params.py --mamba_expand 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import BertConfig, BertForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402

# Baseline kmer1 BERT FM config (matches outputs/bert_mlm/fm_single: 4,939,786 params)
BASE = dict(vocab_size=10, hidden_size=256, num_hidden_layers=6,
            num_attention_heads=4, intermediate_size=1024,
            max_position_embeddings=514, type_vocab_size=1,
            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)


def bert_breakdown(model):
    emb = sum(p.numel() for p in model.bert.embeddings.parameters())
    enc = sum(p.numel() for p in model.bert.encoder.parameters())
    head = sum(p.numel() for p in model.cls.parameters())
    if model.config.tie_word_embeddings:
        head -= model.bert.embeddings.word_embeddings.weight.numel()
    return {"embedding": emb, "encoder": enc, "mlm_head": head,
            "total": sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer_pattern", default="TTMTTM")
    p.add_argument("--mamba_d_state", type=int, default=16)
    p.add_argument("--mamba_d_conv", type=int, default=4)
    p.add_argument("--mamba_expand", type=int, default=4)
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "model_param_compare.json"))
    args = p.parse_args()

    with torch.device("meta") if False else torch.no_grad():
        bert = BertForMaskedLM(BertConfig(**BASE))
        bert_params = bert_breakdown(bert)

        hcfg = HybridMambaConfig(
            **BASE, layer_pattern=args.layer_pattern,
            mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            tie_word_embeddings=args.tie_word_embeddings)
        hybrid = HybridMambaForMaskedLM(hcfg)
        hybrid_params = hybrid.param_groups()

    diff_pct = (hybrid_params["total"] - bert_params["total"]) / bert_params["total"] * 100
    result = {
        "baseline_bert": bert_params,
        "hybrid_mamba": hybrid_params,
        "hybrid_layer_pattern": args.layer_pattern,
        "mamba": {"d_state": args.mamba_d_state, "d_conv": args.mamba_d_conv,
                  "expand": args.mamba_expand,
                  "tie_word_embeddings": args.tie_word_embeddings},
        "total_diff_pct": diff_pct,
        "within_10pct": abs(diff_pct) <= 10.0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{'':12}{'embedding':>12}{'encoder':>12}{'mlm_head':>12}{'total':>12}")
    for name, g in (("BERT", bert_params), ("Hybrid", hybrid_params)):
        print(f"{name:12}{g['embedding']:>12,}{g['encoder']:>12,}"
              f"{g['mlm_head']:>12,}{g['total']:>12,}")
    print(f"\nhybrid pattern : {args.layer_pattern}  (mamba expand={args.mamba_expand}, "
          f"d_state={args.mamba_d_state})")
    print(f"total diff     : {diff_pct:+.2f}%   within ±10%: {result['within_10pct']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
