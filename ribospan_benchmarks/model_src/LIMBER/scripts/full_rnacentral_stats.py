#!/usr/bin/env python3
"""Stream the FULL RNAcentral active FASTA (no filtering, no subsampling) and
report Part-3 statistics: total sequences, total bases, length distribution, and
counts/fractions over the >512 / >1022 / >2048 / >4096 nt thresholds. Also dumps
a length histogram (log-spaced bins) for plotting.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

RAW = Path("data/raw/rnacentral_active.fasta.gz")
OUT = Path("outputs/fm_hybrid_650m"); OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [512, 1022, 2048, 4096]
# histogram bins (nt): fine up to 1022, then coarser
BINS = list(range(0, 1100, 20)) + [1200, 1500, 2048, 3000, 4096, 6000, 10000, 20000, 10**9]


def main():
    t0 = time.time()
    n_seq = 0
    total_bases = 0
    over = {t: 0 for t in THRESHOLDS}
    hist = np.zeros(len(BINS) - 1, dtype=np.int64)
    bins_arr = np.array(BINS)
    maxL = 0
    cur = 0  # current sequence length accumulator

    def flush(L):
        nonlocal n_seq, total_bases, maxL
        if L == 0:
            return
        n_seq += 1
        total_bases += L
        if L > maxL:
            maxL = L
        for t in THRESHOLDS:
            if L > t:
                over[t] += 1
        b = int(np.searchsorted(bins_arr, L, side="right") - 1)
        if 0 <= b < len(hist):
            hist[b] += 1

    with gzip.open(RAW, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                flush(cur)
                cur = 0
                if n_seq and n_seq % 5_000_000 == 0:
                    el = time.time() - t0
                    print(f"  {n_seq:,} seqs, {total_bases:,} bases, {el:.0f}s", flush=True)
            else:
                cur += len(line.strip())
        flush(cur)

    el = time.time() - t0
    stats = {
        "input_file": str(RAW),
        "total_sequences": n_seq,
        "total_bases": total_bases,
        "mean_length": total_bases / max(n_seq, 1),
        "max_length_observed": maxL,
        "over_thresholds": {
            str(t): {"count": over[t], "fraction": over[t] / max(n_seq, 1)}
            for t in THRESHOLDS
        },
        "hist_bins": BINS,
        "hist_counts": hist.tolist(),
        "elapsed_sec": el,
    }
    json.dump(stats, open(OUT / "full_rnacentral_stats.json", "w"), indent=2)
    print(f"\n=== FULL RNAcentral ({el:.0f}s) ===")
    print(f"total sequences : {n_seq:,}")
    print(f"total bases     : {total_bases:,}")
    print(f"mean length     : {total_bases/max(n_seq,1):.1f} nt   max {maxL:,} nt")
    for t in THRESHOLDS:
        print(f"  > {t:>5} nt : {over[t]:>12,}  ({100*over[t]/max(n_seq,1):.3f}%)")
    print(f"\nwrote {OUT}/full_rnacentral_stats.json")


if __name__ == "__main__":
    main()
