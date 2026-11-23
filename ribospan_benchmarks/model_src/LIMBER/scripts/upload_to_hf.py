#!/usr/bin/env python3
"""Upload the trained RNA FM checkpoints to the Hugging Face Hub.

Creates one model repo (default <user>/RNA_FM) and uploads each model under its
own subfolder. Only the final model + tokenizer + config files are uploaded
(checkpoint-*/ and optimizer states are skipped).

Run:
    python scripts/upload_to_hf.py --token hf_xxx --username YOURNAME
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# subfolder -> local model dir (real paths, not symlinks)
MODELS = {
    # ---- LIMBER (TTM hybrid), named as in the paper ------------------------
    "LIMBER-nano":            "outputs/fm_hybrid_100m/epoch_checkpoints/epoch12",
    "LIMBER-meso":            "outputs/fm_hybrid_300m/epoch_checkpoints/epoch12",
    "LIMBER-macro":           "outputs/fm_hybrid_650m/epoch_checkpoints/epoch12",
    "LIMBER-nano-ep6":        "outputs/fm_hybrid_100m/epoch_checkpoints/epoch6",
    "LIMBER-meso-ep6":        "outputs/fm_hybrid_300m/epoch_checkpoints/epoch6",
    "LIMBER-macro-ep6":       "outputs/fm_hybrid_650m/epoch_checkpoints/epoch6",
    "LIMBER-macro-ep9":       "outputs/fm_hybrid_650m/epoch_checkpoints/epoch9",
    "LIMBER-50M":             "outputs/fm_hybrid_50m/epoch_checkpoints/epoch6",
    "LIMBER-10M":             "outputs/hybrid_10m/epoch_checkpoints/epoch6",
    # ---- LIMBER ablations --------------------------------------------------
    "LIMBER-nano-dconv2":     "outputs/fm_hybrid_100m_dconv2",          # conv width 2 (default 4)
    "LIMBER-nano-thin":       "outputs/hybrid_100m_thin/epoch_checkpoints/epoch6",   # 30 x 512
    "LIMBER-nano-wide":       "outputs/hybrid_100m_wide/epoch_checkpoints/epoch6",   # 12 x 768
    "LIMBER-nano-data20pct":  "outputs/hybrid_100m_20pct/epoch_checkpoints/epoch6",
    "LIMBER-nano-data40pct":  "outputs/hybrid_100m_40pct/epoch_checkpoints/epoch6",
    "LIMBER-nano-data60pct":  "outputs/hybrid_100m_60pct/epoch_checkpoints/epoch6",
    "LIMBER-nano-data80pct":  "outputs/hybrid_100m_80pct/epoch_checkpoints/epoch6",
    # ---- size-matched single-architecture controls -------------------------
    "bert_100m_epoch6":       "outputs/bert_100m/epoch_checkpoints/epoch6",
    "bert_300m_epoch6":       "outputs/bert_300m/epoch_checkpoints/epoch6",
    "bert_650m_epoch6":       "outputs/fm_pure_transformer_650m",
    "kmer1-PureMamba-100M":   "outputs/fm_pure_mamba_100m/epoch_checkpoints/epoch6",
    "kmer1-PureMamba-300M":   "outputs/fm_pure_mamba_300m/epoch_checkpoints/epoch6",
    "PureMamba-650M":         "outputs/fm_pure_mamba_650m",
}
# The early k-mer / decoder experiments (kmer1-BERT, kmer6-*, *-decoder) were
# removed from the Hub -- they are not part of the paper. LIMBER-6L256-early and
# the 24x1024 LIMBER-meso-thin are still on the Hub but their local training
# outputs are gone, so they cannot be re-uploaded from here and are not listed.
# only top-level final-model files (bare names -> won't match checkpoint-*/ subdirs)
ALLOW = ["config.json", "model.safetensors", "pytorch_model.bin",
         "model_config.json", "tokenizer_config.json", "vocab.txt",
         "special_tokens_map.json", "tokenizer.json", "generation_config.json",
         "eval_results.json"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="HF write token")
    p.add_argument("--username", required=True)
    p.add_argument("--repo", default="LIMBER")
    p.add_argument("--private", action="store_true")
    p.add_argument("--only", nargs="*", help="upload only these subfolders")
    args = p.parse_args()

    repo_id = f"{args.username}/{args.repo}"
    who = HfApi().whoami(token=args.token).get("name")
    print(f"[hf] token account: {who}  -> repo {repo_id} (private={args.private})")
    create_repo(repo_id, token=args.token, private=args.private,
                repo_type="model", exist_ok=True)

    items = MODELS.items() if not args.only else [(k, MODELS[k]) for k in args.only]
    for sub, rel in items:
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"[hf] SKIP {sub}: {path} missing")
            continue
        print(f"[hf] uploading {sub}  ({rel}) ...")
        upload_folder(repo_id=repo_id, folder_path=str(path), path_in_repo=sub,
                      token=args.token, allow_patterns=ALLOW,
                      commit_message=f"add {sub}")
        print(f"[hf]   done -> https://huggingface.co/{repo_id}/tree/main/{sub}")

    print(f"\n[hf] ALL DONE -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
