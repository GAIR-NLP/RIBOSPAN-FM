#!/usr/bin/env python3
"""Derive non-overlapping k-mer MLM text from the existing single-base files.

Reads train_single.txt / val_single.txt (space-separated bases) and regroups
each sequence into non-overlapping k-mers (tail N-padded to a full k-mer),
writing train_kmer{k}.txt / val_kmer{k}.txt. Reusing the single-base files
guarantees the exact same chunks and train/val split as every other run.

Run:
    python scripts/make_kmer_data.py --k 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLM_DIR = PROJECT_ROOT / "data" / "processed" / "mlm"


def regroup(base_str: str, k: int) -> str:
    toks = []
    for i in range(0, len(base_str), k):
        t = base_str[i:i + k]
        if len(t) < k:
            t = t + "N" * (k - len(t))
        toks.append(t)
    return " ".join(toks)


def convert(src: Path, dst: Path, k: int) -> int:
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            bases = "".join(line.split())
            if not bases:
                continue
            fout.write(regroup(bases, k) + "\n")
            n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--mlm-dir", type=Path, default=MLM_DIR)
    args = p.parse_args(argv)

    for split in ("train", "val"):
        src = args.mlm_dir / f"{split}_single.txt"
        dst = args.mlm_dir / f"{split}_kmer{args.k}.txt"
        n = convert(src, dst, args.k)
        print(f"[kmer-data] k={args.k} {split}: {n:,} seqs -> {dst}")


if __name__ == "__main__":
    main()
