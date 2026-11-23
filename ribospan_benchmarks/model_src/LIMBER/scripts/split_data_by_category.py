#!/usr/bin/env python3
"""
Split rnacentral_active.fasta into stratified subsets (20/40/60/80%) for data
scaling experiments. Each RNA category is split evenly across 5 folds using a
deterministic hash of the URS ID, then folds are combined into cumulative subsets.

Outputs (in DATA_DIR):
  epoch_seqs_20pct/  -- fold 0 only (~20% per category)
  epoch_seqs_40pct/  -- folds 0+1
  epoch_seqs_60pct/  -- folds 0+1+2
  epoch_seqs_80pct/  -- folds 0+1+2+3
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rna_stream_dataset import normalize  # same normalisation as the full corpus
from _paths import PROJ

FASTA   = PROJ / "data/rnacentral/rnacentral_active.fasta"
VAL_TXT = PROJ / "data/rnacentral/epoch_seqs_full/val.txt"
DATA_DIR = PROJ / "data/rnacentral"

FRACS   = [0.2, 0.4, 0.6, 0.8]
NAMES   = ["20pct", "40pct", "60pct", "80pct"]
N_FOLDS = 5
N_EPOCHS = 12  # same as full

def urs_fold(urs_id: str) -> int:
    # md5, not the builtin hash(): string hashing is salted per process unless
    # PYTHONHASHSEED is fixed, so builtin hash() gives different folds on a rerun.
    return int(hashlib.md5(urs_id.encode()).hexdigest(), 16) % N_FOLDS

def fold_included(fold: int, frac_idx: int) -> bool:
    n_include = frac_idx + 1  # frac 0 -> 1 fold, frac 1 -> 2 folds, etc.
    return fold < n_include

def main():
    # Open output files
    out_dirs = []
    out_files = []
    for name in NAMES:
        d = DATA_DIR / f"epoch_seqs_{name}"
        d.mkdir(exist_ok=True)
        out_dirs.append(d)
        out_files.append(open(d / "train.txt", "w"))

    # the held-out validation sequences must never reach a training subset
    val_seqs = {l.strip() for l in open(VAL_TXT) if l.strip()}
    print(f"Loaded {len(val_seqs):,} held-out validation sequences to exclude")

    cat_counts  = defaultdict(int)
    fold_counts = defaultdict(lambda: [0]*N_FOLDS)
    subset_counts = [0] * len(FRACS)
    n_val_skipped = 0

    header = None
    seq_parts = []
    n_total = 0

    def flush(header, seq_parts):
        nonlocal n_total, n_val_skipped
        parts = header.split()
        urs_id   = parts[0][1:]          # strip '>'
        category = parts[1] if len(parts) > 1 else "unknown"
        seq_rna  = normalize("".join(seq_parts))

        if seq_rna in val_seqs:
            n_val_skipped += 1
            return

        fold = urs_fold(urs_id)
        cat_counts[category] += 1
        fold_counts[category][fold] += 1

        for fi, out_f in enumerate(out_files):
            if fold_included(fold, fi):
                out_f.write(seq_rna + "\n")
                subset_counts[fi] += 1
        n_total += 1

        if n_total % 2_000_000 == 0:
            print(f"  processed {n_total:,} sequences...", flush=True)

    print("Parsing FASTA and writing subsets...")
    with open(FASTA) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    flush(header, seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        flush(header, seq_parts)

    for f in out_files:
        f.close()

    print(f"\nTotal sequences processed: {n_total:,}")
    print(f"Validation sequences excluded from every subset: {n_val_skipped:,}")
    print(f"Subset sizes: {dict(zip(NAMES, subset_counts))}")

    # Print category distribution sanity check
    print("\nTop-10 categories (full):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat:<30} {cnt:>10,}  ({cnt/n_total*100:.2f}%)")

    # Create epoch_dir structure for each subset
    n_val = sum(1 for _ in open(VAL_TXT))
    for name, n_train, out_dir in zip(NAMES, subset_counts, out_dirs):
        train_txt = out_dir / "train.txt"
        # symlink epoch files -> train.txt
        for ep in range(1, N_EPOCHS + 1):
            ep_link = out_dir / f"epoch{ep}.txt"
            if ep_link.exists() or ep_link.is_symlink():
                ep_link.unlink()
            ep_link.symlink_to("train.txt")
        # copy val.txt symlink
        val_link = out_dir / "val.txt"
        if val_link.exists() or val_link.is_symlink():
            val_link.unlink()
        val_link.symlink_to(str(VAL_TXT))
        # meta.json
        meta = {"n_train": n_train, "n_val": n_val, "epochs": N_EPOCHS}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\n{name}: {n_train:,} seqs  -> {out_dir}")

    print("\nDone.")

if __name__ == "__main__":
    main()
