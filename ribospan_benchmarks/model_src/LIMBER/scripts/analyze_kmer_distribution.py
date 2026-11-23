#!/usr/bin/env python3
"""K-mer distribution analysis for the RNAcentral active FASTA.

Streams the gzipped FASTA one record at a time (never loading the whole file
into memory) and counts all k-mers (sliding window) for k = 1, 3, 4, 5 over
the first N records. For each k it reports Shannon entropy and effective
vocabulary size, plus the top-20 k-mers.

Run:
    python scripts/analyze_kmer_distribution.py --max-records 100000
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "rnacentral_active.fasta.gz"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rnacentral_stats"

KS = (1, 3, 4, 5)
TOP_N = 20


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


def entropy_bits(counter: Counter, total: int) -> float:
    """Shannon entropy H = -Σ p_i log2(p_i) in bits."""
    if total <= 0:
        return 0.0
    h = 0.0
    for cnt in counter.values():
        p = cnt / total
        h -= p * math.log2(p)
    return h


def main(argv=None):
    p = argparse.ArgumentParser(description="K-mer distribution over RNAcentral FASTA.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--max-records", type=int, default=100_000)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"[error] input not found: {args.input}")

    counters = {k: Counter() for k in KS}
    totals = {k: 0 for k in KS}  # total k-mer windows for each k

    n = 0
    with gzip.open(args.input, "rt") as fh:
        for _header, seq in iter_fasta(fh):
            seq = seq.upper()
            L = len(seq)
            for k in KS:
                if L < k:
                    continue
                c = counters[k]
                # sliding window: ATCG, k=3 -> ATC, TCG
                for i in range(L - k + 1):
                    c[seq[i:i + k]] += 1
                totals[k] += L - k + 1
            n += 1
            if n >= args.max_records:
                break

    result = {}
    for k in KS:
        c = counters[k]
        total = totals[k]
        h = entropy_bits(c, total)
        top = [
            {"kmer": km, "count": cnt, "freq": (cnt / total if total else 0.0)}
            for km, cnt in c.most_common(TOP_N)
        ]
        result[f"{k}mer"] = {
            "entropy": h,
            "effective_vocab_size": (2 ** h),
            "distinct_kmers": len(c),
            "total_kmers": total,
            "top20": top,
        }

        # per-k CSV
        csv_path = args.output_dir / f"top20_{k}mer.csv"
        with open(csv_path, "w") as out:
            out.write("kmer,count,freq\n")
            for row in top:
                out.write(f"{row['kmer']},{row['count']},{row['freq']:.8f}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "kmer_distribution.json"
    with open(json_path, "w") as out:
        json.dump(result, out, indent=2)

    # console report
    print(f"[kmer] input          : {args.input}")
    print(f"[kmer] records         : {n:,}")
    for k in KS:
        r = result[f"{k}mer"]
        print(f"[kmer] k={k}: entropy={r['entropy']:.4f} bits  "
              f"eff_vocab={r['effective_vocab_size']:.1f}  "
              f"distinct={r['distinct_kmers']}  (max 4^{k}={4**k})")
    print(f"[kmer] wrote {json_path}")
    for k in KS:
        print(f"[kmer] wrote {args.output_dir / f'top20_{k}mer.csv'}")


if __name__ == "__main__":
    main()
