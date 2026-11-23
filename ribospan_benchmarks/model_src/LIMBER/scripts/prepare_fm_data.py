#!/usr/bin/env python3
"""Build foundation-model pretraining data: ALL full-dataset sequences whose
base count is < max_length (default 512), with NO chunking (sequences >= the
limit are dropped entirely).

Streams the full RNAcentral active FASTA, normalizes each sequence
(uppercase, T->U, any non-AUCG -> N), keeps min_length <= L < max_length, and
writes parallel train/val text for one or more non-overlapping k-mer schemes
(k=1 is the single-base tokenizer). val split is per-sequence (seeded).

--measure-only just counts (no files written) so you can size the job first.

Run:
    python scripts/prepare_fm_data.py --measure-only
    python scripts/prepare_fm_data.py --output-dir /tmp/rna_fm_data --kmers 1 6
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "rnacentral_active.fasta.gz"
VALID = set("AUCG")


def iter_fasta(handle):
    parts, have = [], False
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if have:
                yield "".join(parts)
            parts, have = [], True
        else:
            parts.append(line)
    if have:
        yield "".join(parts)


def normalize(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in VALID else "N" for c in seq)


def to_kmer(seq: str, k: int) -> str:
    if k == 1:
        return " ".join(seq)
    toks = []
    for i in range(0, len(seq), k):
        t = seq[i:i + k]
        if len(t) < k:
            t = t + "N" * (k - len(t))
        toks.append(t)
    return " ".join(toks)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/rna_fm_data"))
    p.add_argument("--kmers", type=int, nargs="+", default=[1, 6])
    p.add_argument("--min-length", type=int, default=20)
    p.add_argument("--max-length", type=int, default=512)  # keep L < this
    p.add_argument("--val-ratio", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--measure-only", action="store_true")
    p.add_argument("--report-every", type=int, default=5_000_000)
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"[error] input not found: {args.input}")

    rng = random.Random(args.seed)
    writers = {}
    if not args.measure_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for k in args.kmers:
            tag = "single" if k == 1 else f"kmer{k}"
            writers[k] = {
                "train": open(args.output_dir / f"train_{tag}.txt", "w"),
                "val": open(args.output_dir / f"val_{tag}.txt", "w"),
            }

    n_read = n_kept = n_train = n_val = 0
    total_bases = 0
    min_len = None
    max_len = 0
    try:
        with gzip.open(args.input, "rt") as fh:
            for raw in iter_fasta(fh):
                n_read += 1
                if n_read % args.report_every == 0:
                    print(f"[prep] read {n_read:,}  kept {n_kept:,}", flush=True)
                norm = normalize(raw)
                L = len(norm)
                if L < args.min_length or L >= args.max_length:
                    continue
                n_kept += 1
                total_bases += L
                min_len = L if min_len is None else min(min_len, L)
                max_len = max(max_len, L)
                is_val = rng.random() < args.val_ratio
                split = "val" if is_val else "train"
                if is_val:
                    n_val += 1
                else:
                    n_train += 1
                if not args.measure_only:
                    for k in args.kmers:
                        writers[k][split].write(to_kmer(norm, k) + "\n")
    finally:
        for k in writers:
            writers[k]["train"].close()
            writers[k]["val"].close()

    stats = {
        "input_file": str(args.input),
        "records_read": n_read,
        "kept_sequences": n_kept,
        "train_sequences": n_train,
        "val_sequences": n_val,
        "total_bases": total_bases,
        "mean_length": (total_bases / n_kept) if n_kept else 0.0,
        "min_length_observed": min_len or 0,
        "max_length_observed": max_len,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "val_ratio": args.val_ratio,
        "kmers": args.kmers,
        "measure_only": args.measure_only,
    }
    print("\n[prep] " + json.dumps(stats, indent=2))
    if not args.measure_only:
        with open(args.output_dir / "fm_data_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[prep] wrote data to {args.output_dir}/")


if __name__ == "__main__":
    main()
