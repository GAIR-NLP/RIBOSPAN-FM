#!/usr/bin/env python3
"""Pre-sample one random member sequence per mmseqs cluster, independently for
each pretraining epoch -- mirrors RiNALMo's stated training procedure ("sampled
each sequence in the batch from a different cluster"), instead of reusing a
single fixed cluster representative for every epoch.

Input is mmseqs easy-linclust's `*_all_seqs.fasta`: cluster boundaries are
marked by a bare `>cluster_rep_id` line (header only, no sequence) immediately
followed by another header line; every real record is a 2-line
header+single-line-sequence pair. Parsed in one streaming pass (O(1) memory
beyond the current cluster's members).

A fixed set of clusters (val_mod) is held out for validation: always the
cluster's first member (the linclust representative), written once -- never
resampled per epoch, so val loss stays comparable across epochs.

Usage:
  python sample_cluster_epochs.py <all_seqs.fasta> <out_dir> --epochs 6 --val_mod 2048 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

_TRANS = str.maketrans("acgutACGUT", "ACGUUACGUU")


def normalize(seq: str) -> str:
    seq = seq.translate(_TRANS)
    return "".join(c if c in "ACGU" else "N" for c in seq)


def cluster_blocks(path):
    """Yield (cluster_rep_id, [member_seq, ...]) for each cluster, one streaming
    pass with a single-line lookahead."""
    with open(path) as fh:
        line = fh.readline().rstrip("\n")
        cluster_rep = None
        members = []
        while line:
            assert line.startswith(">"), f"expected header, got: {line[:50]!r}"
            nxt = fh.readline().rstrip("\n")
            if nxt.startswith(">") or nxt == "":
                # bare marker (no sequence) -> new cluster boundary
                if cluster_rep is not None and members:
                    yield cluster_rep, members
                cluster_rep = line[1:].split()[0]
                members = []
                line = nxt
                continue
            # real record: line=header, nxt=sequence
            members.append(nxt)
            line = fh.readline().rstrip("\n")
        if cluster_rep is not None and members:
            yield cluster_rep, members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all_seqs")
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--val_mod", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_fhs = [open(out_dir / f"epoch{e + 1}.txt", "w") for e in range(a.epochs)]
    val_fh = open(out_dir / "val.txt", "w")

    t0 = time.time()
    n_clusters = n_val = 0
    for cluster_rep, members in cluster_blocks(a.all_seqs):
        members = [normalize(m) for m in members]
        n_clusters += 1
        is_val = int(hashlib.md5(cluster_rep.encode()).hexdigest(), 16) % a.val_mod == 0
        if is_val:
            val_fh.write(members[0] + "\n")
            n_val += 1
        else:
            for e in range(a.epochs):
                rng = random.Random((a.seed * 1_000_003) ^ (e * 9973) ^ hash(cluster_rep))
                pick = rng.choice(members)
                train_fhs[e].write(pick + "\n")

    for f in train_fhs:
        f.close()
    val_fh.close()

    n_train = n_clusters - n_val
    json.dump({"n_train": n_train, "n_val": n_val, "epochs": a.epochs},
               open(out_dir / "meta.json", "w"), indent=2)
    print(f"[sample] clusters={n_clusters:,} train={n_train:,} val={n_val:,} "
          f"epochs={a.epochs} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
