#!/usr/bin/env python3
"""Prepare MLM training text for the two tokenizers from the RNAcentral FASTA.

Streams the gzipped FASTA one record at a time (never loading the whole file
into memory), normalizes each sequence, chunks long sequences into
non-overlapping pieces of at most --max-length bases, and writes two parallel
text representations of the SAME chunks:

  * single  : space-separated bases        ->  "A U C G A U G G"
  * center3 : space-separated centered 3-mers (N-padded boundaries, token count
              equals base count) -> "NAU AUC UCG CGA GAU AUG UGG GGN"

The train/val split is decided once per chunk (seeded), so both tokenizers see
exactly the same underlying sequences in train and val -> a fair comparison.

Run:
    python scripts/prepare_mlm_data.py --max-records 10000 --max-length 512
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "rnacentral_active.fasta.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "mlm"

VALID = set("AUCG")
# Translation table: uppercase already applied; map T->U, everything handled
# explicitly below.


def iter_fasta(handle):
    """Yield sequence strings one record at a time (headers not needed here)."""
    seq_parts: list[str] = []
    have = False
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if have:
                yield "".join(seq_parts)
            seq_parts = []
            have = True
        else:
            seq_parts.append(line)
    if have:
        yield "".join(seq_parts)


def normalize(seq: str) -> str:
    """Uppercase, T->U, and map any non-AUCG character to N."""
    seq = seq.upper().replace("T", "U")
    return "".join(ch if ch in VALID else "N" for ch in seq)


def chunk_sequence(seq: str, max_length: int):
    """Yield non-overlapping chunks of at most max_length bases."""
    for i in range(0, len(seq), max_length):
        yield seq[i:i + max_length]


def to_single(chunk: str) -> str:
    return " ".join(chunk)


def to_center3(chunk: str) -> str:
    """Centered (overlapping) 3-mers with N-padded boundaries; one token per
    base. NOTE: overlapping tokens share bases with their neighbors, which
    leaks the answer under single-token MLM masking — see to_kmer3 for the
    non-overlapping variant used for a leak-free comparison."""
    padded = "N" + chunk + "N"
    return " ".join(padded[i:i + 3] for i in range(len(chunk)))


def to_kmer3(chunk: str) -> str:
    """Non-overlapping 3-mers (stride 3); the tail is N-padded to a full 3-mer.
    Adjacent tokens share NO bases, so plain single-token MLM masking has no
    neighbor leakage -> directly comparable to the single-base tokenizer."""
    toks = []
    for i in range(0, len(chunk), 3):
        tri = chunk[i:i + 3]
        if len(tri) < 3:
            tri = tri + "N" * (3 - len(tri))
        toks.append(tri)
    return " ".join(toks)


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare MLM data for two tokenizers.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-records", type=int, default=None,
                   help="Limit number of FASTA records read (default: all).")
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--min-length", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"[error] input not found: {args.input}")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    f_train_single = open(args.output_dir / "train_single.txt", "w")
    f_val_single = open(args.output_dir / "val_single.txt", "w")
    f_train_center3 = open(args.output_dir / "train_center3.txt", "w")
    f_val_center3 = open(args.output_dir / "val_center3.txt", "w")
    f_train_kmer3 = open(args.output_dir / "train_kmer3.txt", "w")
    f_val_kmer3 = open(args.output_dir / "val_kmer3.txt", "w")

    n_records = 0
    n_train = 0
    n_val = 0
    train_tok_single = 0
    val_tok_single = 0
    train_tok_center3 = 0
    val_tok_center3 = 0
    train_tok_kmer3 = 0
    val_tok_kmer3 = 0
    total_len = 0
    min_len = None
    max_len = 0

    try:
        with gzip.open(args.input, "rt") as fh:
            for seq in iter_fasta(fh):
                n_records += 1
                norm = normalize(seq)
                for chunk in chunk_sequence(norm, args.max_length):
                    L = len(chunk)
                    if L < args.min_length:
                        continue
                    single_line = to_single(chunk)
                    center3_line = to_center3(chunk)
                    kmer3_line = to_kmer3(chunk)
                    # token counts: single & center3 = L tokens; non-overlapping
                    # kmer3 = ceil(L/3) tokens.
                    n_kmer3 = (L + 2) // 3
                    is_val = rng.random() < args.val_ratio
                    if is_val:
                        f_val_single.write(single_line + "\n")
                        f_val_center3.write(center3_line + "\n")
                        f_val_kmer3.write(kmer3_line + "\n")
                        n_val += 1
                        val_tok_single += L
                        val_tok_center3 += L
                        val_tok_kmer3 += n_kmer3
                    else:
                        f_train_single.write(single_line + "\n")
                        f_train_center3.write(center3_line + "\n")
                        f_train_kmer3.write(kmer3_line + "\n")
                        n_train += 1
                        train_tok_single += L
                        train_tok_center3 += L
                        train_tok_kmer3 += n_kmer3

                    total_len += L
                    min_len = L if min_len is None else min(min_len, L)
                    max_len = max(max_len, L)

                if args.max_records is not None and n_records >= args.max_records:
                    break
    finally:
        for fh in (f_train_single, f_val_single, f_train_center3, f_val_center3,
                   f_train_kmer3, f_val_kmer3):
            fh.close()

    n_chunks = n_train + n_val
    stats = {
        "input_file": str(args.input),
        "records_read": n_records,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_sequences": n_train,
        "val_sequences": n_val,
        "total_sequences": n_chunks,
        "train_tokens_single": train_tok_single,
        "val_tokens_single": val_tok_single,
        "train_tokens_center3": train_tok_center3,
        "val_tokens_center3": val_tok_center3,
        "train_tokens_kmer3": train_tok_kmer3,
        "val_tokens_kmer3": val_tok_kmer3,
        "mean_length": (total_len / n_chunks) if n_chunks else 0.0,
        "max_length_observed": max_len,
        "min_length_observed": min_len if min_len is not None else 0,
    }

    with open(args.output_dir / "data_stats.json", "w") as out:
        json.dump(stats, out, indent=2)

    print(f"[prep] records read     : {n_records:,}")
    print(f"[prep] chunks train/val : {n_train:,} / {n_val:,}")
    print(f"[prep] tokens single t/v: {train_tok_single:,} / {val_tok_single:,}")
    print(f"[prep] tokens center3 t/v: {train_tok_center3:,} / {val_tok_center3:,}")
    print(f"[prep] len min/mean/max : {stats['min_length_observed']} / "
          f"{stats['mean_length']:.1f} / {stats['max_length_observed']}")
    print(f"[prep] wrote {args.output_dir}/")


if __name__ == "__main__":
    main()
