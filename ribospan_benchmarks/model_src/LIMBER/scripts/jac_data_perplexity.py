#!/usr/bin/env python3
"""Full-model (no ablation) masked-LM perplexity on the SAME structured
sequences used for the categorical-Jacobian SS evaluation (bpRNA-derived
PDB/RNP/SRP/tmRNA/tRNAdb/RFAM/CRW, new_inf/bpRNA_1m_90_{name}_less512.cvs),
across every backbone in this study -- our LIMBER-10M/50M/100M/300M/650M and
the four baselines (RiNALMo-giga, AIDO.RNA-650M, RNA-FM, StructRFM).

Same BERT MLM recipe throughout: 15% of nucleotide positions selected,
80% -> [MASK], 10% -> random canonical nt, 10% unchanged; loss only on
selected positions. One fixed-seed pass per sequence (deterministic), no
batching (datasets are small and models/tokenizers are heterogeneous).

Usage:
  python jac_data_perplexity.py --model_type fm_hybrid_650m --out_dir ...
  python jac_data_perplexity.py --model_type structrfm_86m  --out_dir ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from _paths import PROJ

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

DATA_DIR = PROJECT_ROOT / "new_inf"
DATASETS = ["PDB", "RNP", "SRP", "tmRNA", "tRNAdb", "RFAM", "CRW"]
SEED = 42
MLM_P = 0.15

OUR_CKPT_DIRS = {
    "fm_hybrid_10m":  str(PROJECT_ROOT / "outputs/hybrid_10m/epoch_checkpoints/epoch6"),
    "fm_hybrid_50m":  str(PROJ / "checkpoints/fm_hybrid_50m/epoch_checkpoints/epoch6"),
    "fm_hybrid_100m": str(PROJ / "checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch6"),
    "fm_hybrid_300m": str(PROJ / "checkpoints/fm_hybrid_300m/epoch_checkpoints/epoch6"),
    "fm_hybrid_650m": str(PROJ / "checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch6"),
    # epoch12 (final) checkpoints, for the ep6-vs-ep12 comparison
    "fm_hybrid_100m_ep12": str(PROJ / "checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch12"),
    "fm_hybrid_300m_ep12": str(PROJ / "checkpoints/fm_hybrid_300m/epoch_checkpoints/epoch12"),
    "fm_hybrid_650m_ep12": str(PROJ / "checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12"),
    # 100M training-data-fraction ablation, all at epoch6 (the shared,
    # synchronized checkpoint across all four -- see baseline_ppl_epoch6_unified.csv)
    "hybrid_100m_20pct": str(PROJECT_ROOT / "outputs/hybrid_100m_20pct/epoch_checkpoints/epoch6"),
    "hybrid_100m_40pct": str(PROJECT_ROOT / "outputs/hybrid_100m_40pct/epoch_checkpoints/epoch6"),
    "hybrid_100m_60pct": str(PROJECT_ROOT / "outputs/hybrid_100m_60pct/epoch_checkpoints/epoch6"),
    "hybrid_100m_80pct": str(PROJECT_ROOT / "outputs/hybrid_100m_80pct/epoch_checkpoints/epoch6"),
}
EXT_CKPT_DIRS = {
    "rinalmo_giga":  str(PROJ / "checkpoints/rinalmo_giga"),
    "aido_rna_650m": str(PROJ / "checkpoints/aido_rna_650m"),
    "rnafm_100m":    str(PROJ / "checkpoints/rnafm_100m"),
    "structrfm_86m": str(PROJ / "checkpoints/structrfm_86m"),
}
ALL_MODEL_TYPES = list(OUR_CKPT_DIRS) + list(EXT_CKPT_DIRS)
MODEL_NAMES = {
    "fm_hybrid_10m": "LIMBER-10M", "fm_hybrid_50m": "LIMBER-50M",
    "fm_hybrid_100m": "LIMBER-100M-ep6", "fm_hybrid_300m": "LIMBER-300M-ep6",
    "fm_hybrid_650m": "LIMBER-650M-ep6",
    "fm_hybrid_100m_ep12": "LIMBER-100M-ep12", "fm_hybrid_300m_ep12": "LIMBER-300M-ep12",
    "fm_hybrid_650m_ep12": "LIMBER-650M-ep12",
    "hybrid_100m_20pct": "LIMBER-100M-20pct-ep6", "hybrid_100m_40pct": "LIMBER-100M-40pct-ep6",
    "hybrid_100m_60pct": "LIMBER-100M-60pct-ep6", "hybrid_100m_80pct": "LIMBER-100M-80pct-ep6",
    "rinalmo_giga": "RiNALMo-giga",
    "aido_rna_650m": "AIDO.RNA-650M", "rnafm_100m": "RNA-FM",
    "structrfm_86m": "StructRFM",
}

NUC = "AUGC"


# ── Model loading ────────────────────────────────────────────────────────────

def load_ours(ckpt_dir):
    from hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM
    cd = json.load(open(Path(ckpt_dir) / "model_config.json"))
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
    state = torch.load(Path(ckpt_dir) / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    return model


def load_aido_mlm(ckpt_dir):
    """Direct loading (AutoModelForMaskedLM hits the same safetensors-metadata
    bug as AutoModel; see ss_perseq_external.py's _load_aido)."""
    ckpt_dir = Path(ckpt_dir)
    patched = ckpt_dir / "_modeling_aidorna_patched.py"
    if not patched.exists():
        src = (ckpt_dir / "modeling_aidorna.py").read_text()
        src = src.replace("from .configuration_aidorna", "from configuration_aidorna")
        patched.write_text(src)

    sys.path.insert(0, str(ckpt_dir))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_aidorna_model", patched)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AIDORNAForMaskedLM = mod.AIDORNAForMaskedLM

    from configuration_aidorna import AIDORNAConfig
    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    cfg_dict = json.load(open(ckpt_dir / "config.json"))
    exclude = {"_name_or_path", "architectures", "transformers_version",
               "model_type", "auto_map", "torch_dtype"}
    cfg = AIDORNAConfig(**{k: v for k, v in cfg_dict.items() if k not in exclude})
    model = AIDORNAForMaskedLM(cfg)

    # AIDORNAForMaskedLM has self.model = AIDORNAModel(...) and self.cls = ...
    # as direct submodules, so checkpoint keys ("model.xxx", "cls.xxx") map
    # onto it unchanged -- no prefix stripping here (unlike loading the bare
    # AIDORNAModel in ss_perseq_external.py, which does need it stripped).
    sd = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load_aido_mlm] WARNING missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    return model, tok


def load_ext(model_type, ckpt_dir):
    if model_type == "aido_rna_650m":
        return load_aido_mlm(ckpt_dir)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    ckpt_dir = str(ckpt_dir)
    trust = model_type != "structrfm_86m"
    if trust:
        sys.path.insert(0, ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=trust)
    model = AutoModelForMaskedLM.from_pretrained(ckpt_dir, trust_remote_code=trust)
    return model, tok


# ── Masking (generic, tokenizer-agnostic) ───────────────────────────────────

def mask_ours(seq, rng):
    """Returns (ids, labels) tensors, [CLS]+nt+[SEP] layout, PAD/UNK/CLS/SEP/MASK ids fixed."""
    from rna_stream_dataset import CLS as C, SEP as S, MASK as M, NUC as NUCMAP
    ids = [C] + [NUCMAP.get(c, 9) for c in seq] + [S]
    labels = [-100] * len(ids)
    canon_ids = list(NUCMAP[c] for c in NUC)
    for i in range(1, len(ids) - 1):        # skip CLS/SEP
        if rng.random() < MLM_P:
            labels[i] = ids[i]
            r = rng.random()
            if r < 0.8:
                ids[i] = M
            elif r < 0.9:
                ids[i] = rng.choice(canon_ids)
            # else unchanged
    return torch.tensor([ids]), torch.tensor([labels])


def get_canonical_ids(tok):
    ids = []
    for c in NUC:
        enc = tok(c, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            ids.append(enc[0])
    return ids or [tok.unk_token_id]


def mask_ext(seq, tok, has_special, canon_ids, rng):
    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    labels = [-100] * len(ids)
    lo = 1 if has_special else 0
    hi = len(ids) - 1 if has_special else len(ids)
    mask_id = tok.mask_token_id
    for i in range(lo, hi):
        if rng.random() < MLM_P:
            labels[i] = ids[i]
            r = rng.random()
            if r < 0.8:
                ids[i] = mask_id
            elif r < 0.9:
                ids[i] = rng.choice(canon_ids)
    am = enc.get("attention_mask", torch.ones(1, len(ids), dtype=torch.long))
    return torch.tensor([ids]), am, torch.tensor([labels])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", required=True, choices=ALL_MODEL_TYPES)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max_rfam", type=int, default=1000,
                     help="cap RFAM sequences, matching jacobian_ss_eval.py's baseline convention")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ours = args.model_type in OUR_CKPT_DIRS
    print(f"[jac_ppl] model={args.model_type} is_ours={is_ours} device={device}", flush=True)

    if is_ours:
        model = load_ours(OUR_CKPT_DIRS[args.model_type]).to(device).eval()
        tok, has_special, canon_ids = None, True, None
    else:
        model, tok = load_ext(args.model_type, EXT_CKPT_DIRS[args.model_type])
        model = model.to(device).eval()
        has_special = args.model_type != "structrfm_86m"
        canon_ids = get_canonical_ids(tok)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    results = {}

    for ds in DATASETS:
        df = pd.read_csv(DATA_DIR / f"bpRNA_1m_90_{ds}_less512.cvs")
        if ds == "RFAM" and args.max_rfam and len(df) > args.max_rfam:
            df = df.sample(n=args.max_rfam, random_state=args.seed).reset_index(drop=True)
        tot_loss, tot_tok = 0.0, 0
        n_ok = 0
        per_seq = []
        with torch.no_grad():
            for _, row in df.iterrows():
                seq = str(row["seq"]).upper().replace("T", "U")
                if not seq or any(c not in "AUGCN" for c in seq):
                    continue
                if is_ours:
                    ids, labels = mask_ours(seq, rng)
                    ids, labels = ids.to(device), labels.to(device)
                    am = torch.ones_like(ids)
                    h = model.embeddings(input_ids=ids)
                    ext_mask = (1.0 - am[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
                    for layer, is_m in zip(model.layers, model.is_mamba):
                        h = layer(h) if is_m else layer(h, attention_mask=ext_mask)[0]
                    logits = model.cls(h)
                    vocab = model.config.vocab_size
                else:
                    ids, am, labels = mask_ext(seq, tok, has_special, canon_ids, rng)
                    ids, am, labels = ids.to(device), am.to(device), labels.to(device)
                    out = model(input_ids=ids, attention_mask=am)
                    logits = out.logits
                    vocab = logits.shape[-1]

                n_masked = int((labels != -100).sum().item())
                if n_masked == 0:
                    continue
                loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1),
                                        reduction="sum", ignore_index=-100)
                seq_loss = loss.item() / n_masked
                per_seq.append({
                    "id": str(row.get("id", row.get("name", n_ok))),
                    "length": len(seq),
                    "n_masked_tokens": n_masked,
                    "ppl": math.exp(min(seq_loss, 20)),
                })
                tot_loss += loss.item()
                tot_tok += n_masked
                n_ok += 1

        val_loss = tot_loss / max(tot_tok, 1)
        ppl = math.exp(min(val_loss, 20))
        results[ds] = {"val_loss": val_loss, "ppl": ppl, "n_seqs": n_ok, "n_masked_tokens": tot_tok,
                        "per_seq": per_seq}
        print(f"[jac_ppl] {ds}: n={n_ok} PPL={ppl:.4f}", flush=True)

    macro_ppl = sum(r["ppl"] for r in results.values()) / len(results)
    out = {
        "model_type": args.model_type, "model_name": MODEL_NAMES[args.model_type],
        "per_dataset": results, "macro_mean_ppl": macro_ppl,
    }
    out_path = out_dir / "jac_data_ppl.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[jac_ppl] macro_mean_ppl={macro_ppl:.4f}  saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
