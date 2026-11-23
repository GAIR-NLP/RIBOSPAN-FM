#!/usr/bin/env python3
"""Frozen embeddings of the 27 ROBIN RNA targets from several RNA encoders, in the
same target order as rna_target_meta.json. Saves data/robin/emb/<model>_<pool>.npy
(pool = mean over real nucleotides | cls = first token, only for models with a CLS token)
plus a k-mer composition control (no learned encoder)."""
import argparse
import csv
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import downstream_finetune as dft  # noqa: E402
import ss_perseq_external as ext  # noqa: E402
from _paths import PROJ

SEQ_TABLE = PROJECT_ROOT / "data/robin/SMM_Sequence_Table_hit_rates.csv"
OUT = PROJECT_ROOT / "data/robin/emb"
CK = str(PROJ / "checkpoints")
dft.OUR_CKPT_DIRS["fm_pure_transformer_650m"] = f"{CK}/fm_pure_transformer_650m"
dft.OUR_CKPT_DIRS["fm_pure_mamba_650m"] = f"{CK}/fm_pure_mamba_650m"
ALL = ["fm_hybrid_650m", "fm_pure_transformer_650m", "rinalmo_giga", "aido_rna_650m",
       "structrfm_86m", "rnafm_100m"]


def clean_seq(raw):
    s = re.sub(r"\(m\d?([AUGC])\)", r"\1", raw)
    return s.replace(" ", "").replace("+", "").upper().replace("T", "U")


def target_seqs():
    rows = [r for r in csv.DictReader(open(SEQ_TABLE)) if r["Biomolecule Type"] == "RNA"]
    return [r["Nucleic Acid Target"] for r in rows], [clean_seq(r["Sequence (5' to 3')"]) for r in rows]


def kmer_composition(seqs, ks=(1, 2, 3)):
    feats = []
    for s in seqs:
        v = []
        for k in ks:
            kmers = ["".join(p) for p in itertools.product("ACGU", repeat=k)]
            cnt = dict.fromkeys(kmers, 0)
            for i in range(len(s) - k + 1):
                if s[i:i + k] in cnt:
                    cnt[s[i:i + k]] += 1
            tot = max(sum(cnt.values()), 1)
            v += [cnt[m] / tot for m in kmers]
        feats.append(v)
    return np.array(feats, dtype=np.float32)


@torch.no_grad()
def embed(model_type, seqs, device):
    model, tok, _layers, _hid, has_special = dft.load_backbone(model_type, device)
    if model_type == "rinalmo_giga":
        # HF's from_pretrained renames "*.beta" keys, so the trained SwiGLU betas
        # (0.83-1.30) are silently left at their init value 1.0; load them by hand.
        from safetensors import safe_open
        with safe_open(f"{ext.CKPT_DIRS[model_type]}/model.safetensors", "pt") as f:
            for i, layer in enumerate(model.layers):
                layer.ffn.beta.data.copy_(f.get_tensor(f"model.layers.{i}.ffn.beta").to(layer.ffn.beta.dtype))
        print("[emb] rinalmo_giga: restored trained ffn.beta", flush=True)
    ours = model_type in dft.OUR_CKPT_DIRS
    cls_l, mean_l = [], []
    for s in seqs:
        if ours:
            ids, am = dft.tokenize_ours(tok, [s], device)
        else:
            enc = tok([s], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc.get("attention_mask", torch.ones_like(ids)).to(device)
        h = dft.hidden_states(model_type, model, ids, am)[0].float()  # [L, d]
        if has_special:
            mean_l.append(h[1:-1].mean(0).cpu().numpy())
            cls_l.append(h[0].cpu().numpy())
        else:
            mean_l.append(h.mean(0).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return {"mean": np.stack(mean_l), **({"cls": np.stack(cls_l)} if cls_l else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names, seqs = target_seqs()
    print(f"[emb] {len(seqs)} targets: {names[:3]}...", flush=True)
    np.save(OUT / "kmer123_composition.npy", kmer_composition(seqs))
    for m in args.models:
        for pool, e in embed(m, seqs, device).items():
            assert np.isfinite(e).all(), (m, pool)
            np.save(OUT / f"{m}_{pool}.npy", e)
            print(f"[emb] {m:26s} {pool:4s} -> {e.shape}", flush=True)


if __name__ == "__main__":
    main()
