#!/usr/bin/env python3
"""Minimal, self-contained loader for the Hybrid-650M RNA foundation model.

Downloads the checkpoint from the Hugging Face Hub (or uses a local dir),
rebuilds `HybridMambaForMaskedLM` from its `model_config.json`, loads the
weights, and runs a forward pass to get per-nucleotide embeddings + masked-LM
logits for an RNA sequence.

Usage:
    python scripts/load_hybrid650.py                          # download from HF
    python scripts/load_hybrid650.py --model_dir <local_dir>  # use a local dir
    python scripts/load_hybrid650.py --seq ACGUACGUACGU

Requires: torch, transformers==4.46.3, mamba-ssm (see README "Quick start"),
huggingface_hub. The single-nucleotide vocab: PAD0 UNK1 CLS2 SEP3 MASK4 A5 U6 C7 G8 N9.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402

HF_REPO = "Ziyao1010/LIMBER"
HF_SUBFOLDER = "LIMBER-macro"        # 650M, epoch 12 — the model reported in the paper
NUC = {"A": 5, "U": 6, "C": 7, "G": 8, "N": 9}
CLS, SEP = 2, 3
_CFG_KEYS = ("vocab_size", "hidden_size", "num_attention_heads", "intermediate_size",
             "max_position_embeddings", "type_vocab_size", "hidden_dropout_prob",
             "attention_probs_dropout_prob", "pad_token_id", "layer_pattern",
             "mamba_d_state", "mamba_d_conv", "mamba_expand", "tie_word_embeddings")


def load_hybrid(model_dir=None, device="cuda"):
    """Return the loaded HybridMambaForMaskedLM (eval mode)."""
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{HF_SUBFOLDER}/*"])
        model_dir = Path(model_dir) / HF_SUBFOLDER
    model_dir = Path(model_dir)
    cfg = json.load(open(model_dir / "model_config.json"))
    model = HybridMambaForMaskedLM(HybridMambaConfig(**{k: cfg[k] for k in _CFG_KEYS if k in cfg}))
    sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"key mismatch: {missing[:3]} / {unexpected[:3]}"
    return model.to(device).eval(), model_dir


def encode(seq):
    seq = seq.upper().replace("T", "U")
    seq = "".join(c if c in "ACGU" else "N" for c in seq)
    return [CLS] + [NUC[c] for c in seq] + [SEP]


@torch.no_grad()
def embed(model, input_ids, attention_mask=None):
    """Per-nucleotide final-layer representations: [B, L, hidden_size].

    `forward` only returns MLM logits, so run the encoder stack directly -- the
    same path the downstream fine-tuning scripts use.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    h = model.embeddings(input_ids=input_ids, token_type_ids=None)
    ext = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
    for layer, is_m in zip(model.layers, model.is_mamba):
        h = layer(h) if is_m else layer(h, attention_mask=ext)[0]
    return h


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="local dir; default = download from HF")
    ap.add_argument("--seq", default="GGGCUAUUAGCUCAGUUGGUAGAGCGCACCCUUGGUAAGGGUGAGGUCGGCAGUUCGAAUCUGCCUAGCUCCA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, mdir = load_hybrid(args.model_dir, args.device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[load] {mdir}\n[load] HybridMambaForMaskedLM  params={n/1e6:.1f}M  "
          f"layers={model.config.layer_pattern}")

    ids = torch.tensor([encode(args.seq)], device=args.device)
    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=(args.device == "cuda")):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    # per-nucleotide embeddings: strip CLS/SEP. (use the MLM logits' pre-head hidden if you
    # need representations; here we show logits shape + a masked-token prediction demo.)
    logits = out.logits[0]                       # [L+2, vocab]
    print(f"[forward] seq len {len(args.seq)} -> logits {tuple(out.logits.shape)}")
    # demo: mask position 10 and predict
    masked = ids.clone(); masked[0, 10] = 4      # [MASK]
    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=(args.device == "cuda")):
        pred = model(input_ids=masked, attention_mask=torch.ones_like(ids)).logits[0, 10].argmax().item()
    inv = {v: k for k, v in NUC.items()}
    print(f"[demo] masked nt #9 (true {args.seq[9]}) -> predicted {inv.get(pred, pred)}")


if __name__ == "__main__":
    main()
