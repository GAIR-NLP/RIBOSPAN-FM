#!/usr/bin/env python3
"""Prepare full (unclustered, exact-deduped) RNAcentral for pretraining, in the
same epoch_dir/{epoch1..N,val}.txt + meta.json layout that
build_shm_cache_per_epoch() expects -- so it plugs into the existing
per-epoch-aware training pipeline without further code changes. Since there's
no clustering/resampling here, every epoch file is identical (the full
training set, reused every epoch, matching the original single-file design).

Usage: python prep_full_data_epochs.py <dedup.fasta> <out_dir> --epochs 6 --val_mod 2048
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

_TRANS = str.maketrans("acgutACGUT", "ACGUUACGUU")


def normalize(seq: str) -> str:
    seq = seq.translate(_TRANS)
    return "".join(c if c in "ACGU" else "N" for c in seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--val_mod", type=int, default=2048)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_full.txt"
    val_path = out_dir / "val.txt"

    t0 = time.time()
    n_train = n_val = 0
    with open(a.fasta) as fin, open(train_path, "w") as ftr, open(val_path, "w") as fva:
        seq_chunks = []

        def flush():
            nonlocal n_train, n_val
            if not seq_chunks:
                return
            s = normalize("".join(seq_chunks))
            if int(hashlib.md5(s.encode()).hexdigest(), 16) % a.val_mod == 0:
                fva.write(s + "\n")
                n_val += 1
            else:
                ftr.write(s + "\n")
                n_train += 1

        for line in fin:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        flush()

    for e in range(a.epochs):
        link = out_dir / f"epoch{e + 1}.txt"
        link.unlink(missing_ok=True)
        link.symlink_to(train_path.name)

    json.dump({"n_train": n_train, "n_val": n_val, "epochs": a.epochs},
               open(out_dir / "meta.json", "w"), indent=2)
    print(f"[prep] train={n_train:,} val={n_val:,} epochs={a.epochs} "
          f"elapsed={time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
