"""Shared helpers for unsupervised RNA contact probing (kmer1 first version).

Model loading (BERT-MLM / Transformer+Mamba hybrid), kmer1 base-level encoding,
contact-matrix post-processing (symmetrize, diagonal zero, APC correction), the
module map used for embedding-perturbation forward hooks, and the 3-panel
example plot.

kmer6 reserved: base-level contact projection from k-mer tokens needs a separate
design, so these helpers raise on tokenizer_type != "kmer1" for now.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import BertForMaskedLM, BertTokenizerFast

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402

ALPHABET = ["A", "U", "C", "G", "N"]
CANONICAL_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                   ("G", "U"), ("U", "G")}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(model_dir, model_type, device):
    model_dir = Path(model_dir)
    if model_type == "bert":
        model = BertForMaskedLM.from_pretrained(model_dir)
    elif model_type == "hybrid":
        cfg = json.load(open(model_dir / "model_config.json"))
        keep = ("vocab_size", "hidden_size", "num_attention_heads",
                "intermediate_size", "max_position_embeddings", "type_vocab_size",
                "hidden_dropout_prob", "attention_probs_dropout_prob", "pad_token_id",
                "layer_pattern", "mamba_d_state", "mamba_d_conv", "mamba_expand",
                "tie_word_embeddings")
        model = HybridMambaForMaskedLM(HybridMambaConfig(**{k: cfg[k] for k in keep if k in cfg}))
        bin_path = model_dir / "pytorch_model.bin"
        if bin_path.exists():
            sd = torch.load(bin_path, map_location="cpu")
        else:
            from safetensors.torch import load_file
            sd = load_file(str(model_dir / "model.safetensors"))
        model.load_state_dict(sd, strict=False)
    else:
        raise ValueError(f"unknown model_type {model_type!r}")
    return model.to(device).eval()


def load_tokenizer(vocab_dir):
    return BertTokenizerFast(vocab_file=str(Path(vocab_dir) / "vocab.txt"),
                             do_lower_case=False)


def base_token_ids(tokenizer):
    return [tokenizer.convert_tokens_to_ids(b) for b in ALPHABET]


def encode_kmer1(tokenizer, seq, device):
    """base string -> input_ids [1, L+2] = [CLS] b0..b_{L-1} [SEP]."""
    ids = [tokenizer.cls_token_id] + \
          [tokenizer.convert_tokens_to_ids(c) for c in seq] + \
          [tokenizer.sep_token_id]
    t = torch.tensor([ids], device=device)
    am = torch.ones_like(t)
    return t, am


def module_map(model, model_type):
    """Modules for embedding-perturbation hooks: the embedding module and the
    ordered list of encoder-layer modules (for perturb_layer selection)."""
    if model_type == "bert":
        return {"embedding": model.bert.embeddings,
                "layers": list(model.bert.encoder.layer)}
    if model_type == "hybrid":
        return {"embedding": model.embeddings, "layers": list(model.layers)}
    raise ValueError(model_type)


# --------------------------------------------------------------------------- #
# Contact-matrix post-processing
# --------------------------------------------------------------------------- #
def symmetrize_zero_diag(S):
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 0.0)
    return S


def apc(S):
    """Average Product Correction: S_apc[i,j] = S[i,j] - r_i c_j / mean."""
    S = S.copy()
    n = S.shape[0]
    row = S.mean(axis=1, keepdims=True)
    col = S.mean(axis=0, keepdims=True)
    g = S.mean()
    if g == 0:
        return S
    return S - (row @ col) / g


def finalize(S_raw):
    """Return (S_sym, S_apc): symmetrized/zero-diag and its APC correction."""
    S_sym = symmetrize_zero_diag(np.asarray(S_raw, dtype=np.float64))
    return S_sym, apc(S_sym)


# --------------------------------------------------------------------------- #
# Plotting (3 panels: gold | pred pre-APC | pred post-APC)
# --------------------------------------------------------------------------- #
def gold_matrix(length, pairs):
    G = np.zeros((length, length), dtype=np.float32)
    for i, j in pairs:
        G[i, j] = G[j, i] = 1.0
    return G


def plot_example(out_png, rec, S_sym, S_apc, auprc=None, f1=None):
    L = rec["length"]
    G = gold_matrix(L, rec["pairs"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, M, t in ((axes[0], G, "gold contacts"),
                     (axes[1], S_sym, "predicted (pre-APC)"),
                     (axes[2], S_apc, "predicted (post-APC)")):
        im = ax.imshow(M, cmap="viridis", origin="upper")
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    extra = []
    if auprc is not None:
        extra.append(f"AUPRC={auprc:.3f}")
    if f1 is not None:
        extra.append(f"F1={f1:.3f}")
    fig.suptitle(f"{rec['id']}  | family={rec['family']} | L={L}  "
                 + "  ".join(extra), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
