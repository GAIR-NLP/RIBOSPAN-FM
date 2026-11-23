#!/usr/bin/env python3
"""Streaming analysis pipeline for the RNAcentral active sequences FASTA.

Reads a FASTA / FASTA.gz file one record at a time (never loading the whole
file into memory) and produces summary statistics, length histograms,
alphabet counts and a small sample of records.

If no --input is given and the default raw file does not exist, the script
downloads it from the EBI RNAcentral FTP server first.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional for the core logic
    tqdm = None


# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "rnacentral_active.fasta.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rnacentral_stats"
DOWNLOAD_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/"
    "sequences/rnacentral_active.fasta.gz"
)

# Canonical alphabet we track explicitly; everything else is "other".
# The RNAcentral *active* FASTA stores sequences in DNA alphabet (T), while
# other RNA sources use U, so we track both T and U.
CANONICAL = ("A", "U", "T", "C", "G", "N")
# Ambiguous = anything that is not a definite single base A/U/T/C/G (includes N
# and IUPAC ambiguity codes such as R, Y, S, W, K, M, B, D, H, V).
DEFINITE_BASES = frozenset("AUTCG")

SAMPLE_LIMIT = 100
TOP_N_LONGEST = 20
TOP_N_TAXIDS = 20


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def ensure_input(input_path: Path) -> Path:
    """Return a usable input path, downloading the default file if missing."""
    if input_path.exists():
        return input_path

    # Only auto-download the default raw file; a user-specified missing path
    # is treated as an error.
    if input_path.resolve() != DEFAULT_INPUT.resolve():
        sys.exit(f"[error] input file not found: {input_path}")

    input_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {DOWNLOAD_URL}")
    print(f"[download] -> {input_path}")
    tmp_path = input_path.with_suffix(input_path.suffix + ".part")

    def _report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done / total_size * 100
        sys.stdout.write(
            f"\r[download] {done/1e9:.2f} / {total_size/1e9:.2f} GB ({pct:5.1f}%)"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tmp_path, reporthook=_report)
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        sys.exit(f"\n[error] download failed: {exc}")
    sys.stdout.write("\n")
    tmp_path.rename(input_path)
    print(f"[download] done ({input_path.stat().st_size/1e9:.2f} GB)")
    return input_path


# --------------------------------------------------------------------------- #
# Streaming FASTA reader
# --------------------------------------------------------------------------- #
def open_fasta(path: Path):
    """Open a FASTA / .fa / .fasta.gz file as a text stream."""
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rt")
    if name.endswith((".fasta", ".fa", ".fna", ".txt")):
        return open(path, "rt")
    # Fall back to plain text; many RNAcentral mirrors keep .fasta naming.
    return open(path, "rt")


def iter_fasta(handle):
    """Yield (header, sequence) tuples one record at a time.

    The sequence for each record is accumulated from its lines, but only one
    record is ever held in memory at a time.
    """
    header = None
    seq_parts: list[str] = []
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_parts)
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def parse_taxid(header: str) -> str | None:
    """Extract taxonomy id from a header like 'URS00005EB5B7_9606 ...'.

    Returns the taxid string (e.g. '9606') or None if not present.
    """
    first_token = header.split(None, 1)[0]
    if "_" in first_token:
        taxid = first_token.rsplit("_", 1)[1]
        if taxid.isdigit():
            return taxid
    return None


def parse_id(header: str) -> str:
    """Return the record id (first whitespace-delimited token)."""
    return header.split(None, 1)[0]


# --------------------------------------------------------------------------- #
# Percentile helper (operates on a sorted list)
# --------------------------------------------------------------------------- #
def percentile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100]) over a sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_vals[low])
    frac = rank - low
    return sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def analyze(input_path: Path, output_dir: Path, max_records: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_seqs = 0
    total_nt = 0
    lengths: list[int] = []
    alphabet = Counter()           # per-character counts across all sequences
    ambiguous_nt = 0               # nucleotides that are not a definite A/U/C/G
    taxid_counter = Counter()
    seq_hash_counter = Counter()   # md5 -> count, for exact-duplicate detection

    # Maintain top-N longest without storing every sequence.
    longest: list[tuple[int, str, str | None]] = []  # (length, id, taxid)

    sample_path = output_dir / "sample_records.jsonl"
    sample_count = 0

    progress = None
    if tqdm is not None:
        progress = tqdm(unit=" seq", desc="analyzing")

    with open_fasta(input_path) as handle, open(sample_path, "w") as sample_fh:
        for header, seq in iter_fasta(handle):
            seq = seq.upper()
            length = len(seq)
            rec_id = parse_id(header)
            taxid = parse_taxid(header)

            n_seqs += 1
            total_nt += length
            lengths.append(length)

            # Alphabet + ambiguous counting in a single pass over the seq.
            local = Counter(seq)
            for ch, cnt in local.items():
                alphabet[ch] += cnt
                if ch not in DEFINITE_BASES:
                    ambiguous_nt += cnt

            if taxid is not None:
                taxid_counter[taxid] += 1

            # Exact-duplicate detection via md5 of the sequence.
            digest = hashlib.md5(seq.encode("ascii", "ignore")).hexdigest()
            seq_hash_counter[digest] += 1

            # Track top-N longest sequences.
            if len(longest) < TOP_N_LONGEST:
                longest.append((length, rec_id, taxid))
                longest.sort(key=lambda x: x[0])
            elif length > longest[0][0]:
                longest[0] = (length, rec_id, taxid)
                longest.sort(key=lambda x: x[0])

            # Save first SAMPLE_LIMIT records.
            if sample_count < SAMPLE_LIMIT:
                json.dump(
                    {
                        "id": rec_id,
                        "taxid": taxid,
                        "length": length,
                        "sequence_prefix": seq[:50],
                    },
                    sample_fh,
                )
                sample_fh.write("\n")
                sample_count += 1

            if progress is not None:
                progress.update(1)

            if max_records is not None and n_seqs >= max_records:
                break

    if progress is not None:
        progress.close()

    # ------------------------------------------------------------------ #
    # Derived statistics
    # ------------------------------------------------------------------ #
    lengths_sorted = sorted(lengths)
    if lengths_sorted:
        length_stats = {
            "min": lengths_sorted[0],
            "max": lengths_sorted[-1],
            "mean": sum(lengths_sorted) / len(lengths_sorted),
            "median": percentile(lengths_sorted, 50),
            "p50": percentile(lengths_sorted, 50),
            "p75": percentile(lengths_sorted, 75),
            "p90": percentile(lengths_sorted, 90),
            "p95": percentile(lengths_sorted, 95),
            "p99": percentile(lengths_sorted, 99),
        }
    else:
        length_stats = {k: 0 for k in
                        ("min", "max", "mean", "median",
                         "p50", "p75", "p90", "p95", "p99")}

    # Alphabet summary: canonical bases + "other".
    other_count = 0
    alphabet_summary = {b: int(alphabet.get(b, 0)) for b in CANONICAL}
    for ch, cnt in alphabet.items():
        if ch not in CANONICAL:
            other_count += cnt
    alphabet_summary["other"] = int(other_count)

    ambiguous_ratio = (ambiguous_nt / total_nt) if total_nt else 0.0

    # Duplicate exact sequences: number of records that are a repeat of a
    # sequence already seen (i.e. total records minus unique sequences).
    unique_sequences = len(seq_hash_counter)
    duplicate_count = n_seqs - unique_sequences

    top_longest = [
        {"id": rid, "taxid": tid, "length": ln}
        for ln, rid, tid in sorted(longest, key=lambda x: x[0], reverse=True)
    ]

    top_taxids = [
        {"taxid": tid, "count": cnt}
        for tid, cnt in taxid_counter.most_common(TOP_N_TAXIDS)
    ]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "max_records": max_records,
        "sequence_count": n_seqs,
        "nucleotide_total": total_nt,
        "length_stats": length_stats,
        "alphabet_counts": alphabet_summary,
        "ambiguous_nucleotide_ratio": ambiguous_ratio,
        "duplicate_exact_sequence_count": duplicate_count,
        "unique_sequence_count": unique_sequences,
        "top_longest_sequences": top_longest,
        "top_taxonomy_ids": top_taxids,
        "sample_records_file": str(sample_path),
    }

    # ------------------------------------------------------------------ #
    # Write outputs
    # ------------------------------------------------------------------ #
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    write_length_hist(lengths_sorted, output_dir / "length_hist.csv")
    write_alphabet_csv(alphabet_summary, output_dir / "alphabet_counts.csv")

    return summary


def write_length_hist(lengths_sorted: list[int], path: Path, n_bins: int = 50) -> None:
    """Write a length histogram as bin_start,bin_end,count."""
    with open(path, "w") as fh:
        fh.write("bin_start,bin_end,count\n")
        if not lengths_sorted:
            return
        lo, hi = lengths_sorted[0], lengths_sorted[-1]
        if lo == hi:
            fh.write(f"{lo},{hi},{len(lengths_sorted)}\n")
            return
        width = (hi - lo) / n_bins
        counts = [0] * n_bins
        for v in lengths_sorted:
            idx = int((v - lo) / width)
            if idx >= n_bins:
                idx = n_bins - 1
            counts[idx] += 1
        for i, c in enumerate(counts):
            bin_start = lo + i * width
            bin_end = lo + (i + 1) * width
            fh.write(f"{bin_start:.2f},{bin_end:.2f},{c}\n")


def write_alphabet_csv(alphabet_summary: dict, path: Path) -> None:
    with open(path, "w") as fh:
        fh.write("symbol,count\n")
        for sym in (*CANONICAL, "other"):
            fh.write(f"{sym},{alphabet_summary.get(sym, 0)}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Streaming statistics for an RNAcentral FASTA file."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to a .fasta / .fa / .fasta.gz file "
             "(default: data/raw/rnacentral_active.fasta.gz, auto-downloaded).",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after this many sequences (for quick smoke tests).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: outputs/rnacentral_stats).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = ensure_input(args.input)
    print(f"[analyze] input    : {input_path}")
    print(f"[analyze] output   : {args.output_dir}")
    print(f"[analyze] max_recs : {args.max_records}")

    summary = analyze(input_path, args.output_dir, args.max_records)

    print("\n[done] summary:")
    print(f"  sequence_count          : {summary['sequence_count']:,}")
    print(f"  nucleotide_total        : {summary['nucleotide_total']:,}")
    ls = summary["length_stats"]
    print(f"  length min/mean/max     : {ls['min']} / {ls['mean']:.1f} / {ls['max']}")
    print(f"  ambiguous_nt_ratio      : {summary['ambiguous_nucleotide_ratio']:.6f}")
    print(f"  duplicate_exact_seqs    : {summary['duplicate_exact_sequence_count']:,}")
    print(f"  distinct_taxids (top20) : {len(summary['top_taxonomy_ids'])}")
    print(f"\n[outputs] {args.output_dir}")
    for name in ("summary.json", "length_hist.csv",
                 "alphabet_counts.csv", "sample_records.jsonl"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
