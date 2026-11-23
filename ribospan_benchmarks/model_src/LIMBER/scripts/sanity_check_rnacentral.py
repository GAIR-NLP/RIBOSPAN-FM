#!/usr/bin/env python3
"""Lightweight sanity check for the RNAcentral active FASTA.

Streams the gzipped FASTA one record at a time (never loading the whole file
into memory), checks the first N records (default 100,000), prints a short
report plus warnings, and writes outputs/rnacentral_stats/sanity_check.json.

Run:
    python scripts/sanity_check_rnacentral.py --max-records 100000
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "rnacentral_active.fasta.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "rnacentral_stats" / "sanity_check.json"

CANONICAL = ("A", "T", "U", "C", "G", "N")  # tracked explicitly
SHORT_LEN = 10        # length < 10 is suspiciously short
LONG_LEN = 100_000    # length > 100000 is suspiciously long


def iter_fasta(handle):
    """Yield (header, sequence) one record at a time."""
    header = None
    parts: list[str] = []
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(parts)
            header = line[1:].strip()
            parts = []
        else:
            parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def header_ok(header: str) -> bool:
    """A header parses if it has a non-empty first token (the accession)."""
    if not header:
        return False
    first = header.split(None, 1)[0]
    return bool(first)


def main(argv=None):
    p = argparse.ArgumentParser(description="Sanity check the RNAcentral FASTA.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--max-records", type=int, default=100_000)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"[error] input not found: {args.input}")

    n = 0
    total_nt = 0
    min_len = None
    max_len = 0
    alphabet = Counter()       # every character seen
    empty_seqs = 0
    header_fail = 0
    short_seqs = 0
    long_seqs = 0
    samples: list[dict] = []

    with gzip.open(args.input, "rt") as fh:
        for header, seq in iter_fasta(fh):
            seq = seq.upper()
            length = len(seq)

            n += 1
            total_nt += length
            if length == 0:
                empty_seqs += 1
            else:
                min_len = length if min_len is None else min(min_len, length)
                max_len = max(max_len, length)
            if length and length < SHORT_LEN:
                short_seqs += 1
            if length > LONG_LEN:
                long_seqs += 1

            if not header_ok(header):
                header_fail += 1

            alphabet.update(seq)

            if len(samples) < 5:
                samples.append({
                    "header": header,
                    "length": length,
                    "sequence_prefix": seq[:60],
                })

            if n >= args.max_records:
                break

    # Alphabet summary: canonical bases + "other".
    alpha_summary = {b: int(alphabet.get(b, 0)) for b in CANONICAL}
    other_chars = {ch: int(cnt) for ch, cnt in alphabet.items()
                   if ch not in CANONICAL}
    other_total = sum(other_chars.values())
    alpha_summary["other"] = other_total

    mean_len = (total_nt / n) if n else 0.0
    other_ratio = (other_total / total_nt) if total_nt else 0.0

    # ------------------------------------------------------------------ #
    # Warnings
    # ------------------------------------------------------------------ #
    warnings = []
    u_cnt = alpha_summary["U"]
    t_cnt = alpha_summary["T"]
    if u_cnt == 0 or (t_cnt > 0 and u_cnt < t_cnt * 0.01):
        warnings.append(
            f"DNA alphabet detected: U={u_cnt} (≈0), T={t_cnt} (many) — "
            "sequences are stored with T, not U (expected for the active set)."
        )
    if other_ratio > 0.01:
        warnings.append(
            f"'other' character ratio is {other_ratio:.4%} (> 1%) — "
            f"unexpectedly many non-ATUCGN characters: {other_chars}"
        )
    if empty_seqs > 0:
        warnings.append(f"{empty_seqs} empty sequence(s) found.")
    if short_seqs > 0:
        warnings.append(f"{short_seqs} sequence(s) shorter than {SHORT_LEN} nt.")
    if long_seqs > 0:
        warnings.append(f"{long_seqs} sequence(s) longer than {LONG_LEN} nt.")
    if header_fail > 0:
        warnings.append(f"{header_fail} header(s) failed to parse.")

    result = {
        "input_file": str(args.input),
        "max_records": args.max_records,
        "sequence_count": n,
        "total_nt": total_nt,
        "min_len": min_len if min_len is not None else 0,
        "mean_len": mean_len,
        "max_len": max_len,
        "alphabet_counts": alpha_summary,
        "other_characters": other_chars,
        "empty_sequence_count": empty_seqs,
        "header_parse_failures": header_fail,
        "short_sequence_count_lt10": short_seqs,
        "long_sequence_count_gt100000": long_seqs,
        "sample_records": samples,
        "warnings": warnings,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        json.dump(result, out, indent=2)

    # ------------------------------------------------------------------ #
    # Console report
    # ------------------------------------------------------------------ #
    print(f"[sanity] input            : {args.input}")
    print(f"[sanity] sequence_count   : {n:,}")
    print(f"[sanity] total_nt         : {total_nt:,}")
    print(f"[sanity] len min/mean/max : {result['min_len']} / {mean_len:.1f} / {max_len}")
    print(f"[sanity] alphabet         : {alpha_summary}")
    print(f"[sanity] other chars      : {other_chars}")
    print(f"[sanity] empty seqs       : {empty_seqs}")
    print(f"[sanity] header failures  : {header_fail}")
    print(f"[sanity] short (<10)      : {short_seqs}")
    print(f"[sanity] long (>100000)   : {long_seqs}")
    print("[sanity] first 5 samples:")
    for s in samples:
        print(f"   > {s['header'][:70]}  (len={s['length']})")
        print(f"     {s['sequence_prefix']}")
    if warnings:
        print("\n[sanity] WARNINGS:")
        for w in warnings:
            print(f"   ! {w}")
    else:
        print("\n[sanity] no warnings.")
    print(f"\n[sanity] wrote {args.output}")


if __name__ == "__main__":
    main()
