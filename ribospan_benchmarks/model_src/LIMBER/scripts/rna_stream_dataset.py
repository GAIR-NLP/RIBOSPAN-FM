#!/usr/bin/env python3
"""Streaming dataset for full-RNAcentral MLM pretraining (Hybrid-650M).

Design (matches the task spec):
  * FULL RNAcentral, no subsampling / no length filter / no family filter.
  * The 9 GB gz is decompressed ONCE into /dev/shm as one normalized sequence
    per line (RAM tmpfs, no disk use); training streams from there.
  * Context length 1022. L<=1022 -> full sequence. L>1022 -> RANDOM crop
    start ~ Uniform(0, L-1022), re-sampled every epoch (different window per
    epoch; never fixed-front / fixed-center / cached).
  * Normalization: upper-case, T->U, anything not in {A,U,C,G} -> N.
  * MLM: BERT 15%, of which 80% -> [MASK], 10% -> random nucleotide, 10% kept;
    loss only on masked positions.

Vocab (tokenizers/single): PAD0 UNK1 CLS2 SEP3 MASK4  A5 U6 C7 G8 N9.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

# vocab ids (tokenizers/single)
PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
NUC = {"A": 5, "U": 6, "C": 7, "G": 8, "N": 9}
CANON_IDS = [5, 6, 7, 8]              # random-replacement pool ("random nucleotide")
_TRANS = str.maketrans("acgutACGUT", "ACGUUACGUU")   # lower->upper, T/t->U


def normalize(seq: str) -> str:
    seq = seq.translate(_TRANS)
    return "".join(c if c in "ACGU" else "N" for c in seq)


def build_shm_cache(gz_path, shm_path, val_path, val_mod=2048, min_len=1, log=print):
    """Decompress + normalize RNAcentral into /dev/shm (one seq/line), holding out
    a deterministic validation split (md5(seq) % val_mod == 0) written to val_path.
    Idempotent: skips if shm_path already complete (marker file)."""
    shm_path, val_path = Path(shm_path), Path(val_path)
    marker = Path(str(shm_path) + ".done")
    if marker.exists() and shm_path.exists():
        n_train = int(marker.read_text().split()[0])
        log(f"[cache] reuse {shm_path} ({n_train:,} train seqs)")
        return n_train
    val_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_train = n_val = 0
    with gzip.open(gz_path, "rt") as fh, open(shm_path, "w") as ftr, open(val_path, "w") as fva:
        seq_lines = []

        def flush():
            nonlocal n_train, n_val
            if not seq_lines:
                return
            s = normalize("".join(seq_lines))
            if len(s) < min_len:
                return
            if int(hashlib.md5(s.encode()).hexdigest(), 16) % val_mod == 0:
                fva.write(s + "\n"); n_val += 1
            else:
                ftr.write(s + "\n"); n_train += 1

        for line in fh:
            if line.startswith(">"):
                flush(); seq_lines = []
            else:
                seq_lines.append(line.strip())
        flush()
    marker.write_text(f"{n_train} {n_val}\n")
    log(f"[cache] wrote {n_train:,} train + {n_val:,} val seqs in {time.time()-t0:.0f}s -> {shm_path}")
    return n_train


def epoch_cache_fingerprint(epoch_dir, meta):
    """Fingerprint tying a /dev/shm cache to *this* epoch_dir + n_train/n_val --
    a stale cache from a different run (e.g. left over on a recycled node)
    must never be silently reused."""
    return f"{Path(epoch_dir).resolve()}|{meta['n_train']}|{meta['n_val']}|{meta['epochs']}"


def epoch_cache_is_fresh(epoch_dir, shm_dir, meta, fingerprint=None):
    """Read-only freshness check, safe to call from every rank concurrently
    (no file mutation) -- used to gate which rank performs the rebuild."""
    shm_dir = Path(shm_dir)
    fingerprint = fingerprint or epoch_cache_fingerprint(epoch_dir, meta)
    marker = shm_dir / "rna_seqs_per_epoch.done"
    epoch_paths = [shm_dir / f"rna_seqs_epoch{e + 1}.txt" for e in range(meta["epochs"])]
    val_shm = shm_dir / "rna_seqs_val.txt"
    return (marker.exists() and marker.read_text() == fingerprint
            and all(p.exists() for p in epoch_paths) and val_shm.exists())


def build_shm_cache_per_epoch(epoch_dir, shm_dir, log=print):
    """Copy pre-sampled per-epoch cluster files (scripts/sample_cluster_epochs.py
    output: epoch_dir/{epoch1..N,val}.txt + meta.json) into /dev/shm so training
    reads from tmpfs. Idempotent: skips if shm marker is already complete and
    fresh. NOT safe to call concurrently from multiple ranks when the cache is
    stale/missing -- callers must gate the rebuild to a single rank (see
    epoch_cache_is_fresh) and have other ranks just wait, since the purge+copy
    below is not atomic across processes.
    Returns (n_train, val_shm_path, [epoch_shm_path, ...])."""
    import json
    import shutil

    epoch_dir, shm_dir = Path(epoch_dir), Path(shm_dir)
    meta = json.load(open(epoch_dir / "meta.json"))
    n_epochs = meta["epochs"]
    fingerprint = epoch_cache_fingerprint(epoch_dir, meta)
    marker = shm_dir / "rna_seqs_per_epoch.done"
    epoch_paths = [shm_dir / f"rna_seqs_epoch{e + 1}.txt" for e in range(n_epochs)]
    val_shm = shm_dir / "rna_seqs_val.txt"
    if epoch_cache_is_fresh(epoch_dir, shm_dir, meta, fingerprint):
        log(f"[cache] reuse per-epoch shm cache ({meta['n_train']:,} train seqs/epoch)")
        return meta["n_train"], val_shm, epoch_paths
    if marker.exists():
        log("[cache] stale /dev/shm cache from a different run found -- purging")
        for p in epoch_paths + [val_shm, marker]:
            p.unlink(missing_ok=True)
        for p in shm_dir.glob("rna_seqs_epoch*.txt"):
            p.unlink(missing_ok=True)

    shm_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # epoch{N}.txt may be symlinks to the same underlying file (e.g. full-data
    # mode with no per-epoch resampling) -- copy each *distinct* source once
    # and symlink the rest, so identical epochs don't multiply RAM usage.
    copied = {}
    for e, dst in enumerate(epoch_paths):
        src = (epoch_dir / f"epoch{e + 1}.txt").resolve()
        if src in copied:
            dst.unlink(missing_ok=True)
            dst.symlink_to(copied[src])
        else:
            shutil.copy2(src, dst)
            copied[src] = dst
    shutil.copy2(epoch_dir / "val.txt", val_shm)
    marker.write_text(fingerprint)
    log(f"[cache] copied {n_epochs} per-epoch files "
        f"({meta['n_train']:,} train seqs/epoch) in {time.time() - t0:.0f}s -> {shm_dir}")
    return meta["n_train"], val_shm, epoch_paths


def encode(seq, max_len, rng):
    """Random-crop (if needed) + add CLS/SEP. Returns (ids, special_mask)."""
    L = len(seq)
    if L > max_len:
        start = rng.randint(0, L - max_len)       # Uniform(0, L-max_len) inclusive
        seq = seq[start:start + max_len]
    ids = [CLS] + [NUC.get(c, UNK) for c in seq] + [SEP]
    spec = [1] + [0] * (len(seq)) + [1]
    return ids, spec


class RNAStreamDataset(IterableDataset):
    """Infinite stream over the /dev/shm one-seq-per-line cache, sharded by
    (rank x dataloader-worker), shuffle-buffered, with per-epoch random crops."""

    def __init__(self, shm_path, max_len=1022, shuffle_buffer=50000, seed=42,
                 group_size=0, start_epoch=0):
        """shm_path: a single file (reused every epoch) or a list of files,
        one per epoch, cycled by epoch index (RiNALMo-style dynamic per-epoch
        cluster resampling -- see sample_cluster_epochs.py).
        start_epoch: initial epoch offset, so a resumed run (continuing past
        N already-trained epochs) picks up at shm_paths[N] instead of
        replaying file 0 -- each process's internal epoch counter otherwise
        always starts at 0, independent of the Trainer's global_step."""
        self.shm_paths = [str(shm_path)] if isinstance(shm_path, (str, Path)) else [str(p) for p in shm_path]
        self.max_len = max_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.start_epoch = start_epoch
        # group_size>0 -> length-grouped batching: buffer is sorted by post-crop
        # length and emitted in homogeneous chunks of group_size (shuffled order),
        # so the DataLoader's batches contain similar-length seqs => minimal
        # padding. group_size MUST equal the per-device batch size.
        self.group_size = group_size

    def _shard(self):
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1
        return rank * nworkers + wid, world * nworkers

    def _emit_grouped(self, buf, rng):
        """Sort the buffered (ids,spec) by length, chunk into length-homogeneous
        groups of group_size, shuffle group order, and yield example by example."""
        buf.sort(key=lambda x: len(x[0]))
        gs = self.group_size
        groups = [buf[i:i + gs] for i in range(0, len(buf), gs)]
        rng.shuffle(groups)
        for g in groups:
            for ids, spec in g:
                yield {"input_ids": ids, "special_tokens_mask": spec}

    def __iter__(self):
        shard_id, n_shards = self._shard()
        epoch = self.start_epoch
        while True:                                   # infinite: one pass = one epoch
            rng = random.Random((self.seed * 1_000_003) ^ (epoch * 9973) ^ (shard_id * 131))
            buf = []
            shm_path = self.shm_paths[epoch % len(self.shm_paths)]
            with open(shm_path) as fh:
                for idx, line in enumerate(fh):
                    if idx % n_shards != shard_id:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    if self.group_size:               # length-grouped batching
                        buf.append(encode(line, self.max_len, rng))
                        if len(buf) >= self.shuffle_buffer:
                            yield from self._emit_grouped(buf, rng); buf = []
                        continue
                    # plain shuffle-buffer (single examples)
                    if len(buf) < self.shuffle_buffer:
                        buf.append(line); continue
                    j = rng.randrange(len(buf))
                    pick = buf[j]; buf[j] = line
                    ids, spec = encode(pick, self.max_len, rng)
                    yield {"input_ids": ids, "special_tokens_mask": spec}
            if self.group_size:
                if buf:
                    yield from self._emit_grouped(buf, rng)
            else:
                rng.shuffle(buf)
                for pick in buf:
                    ids, spec = encode(pick, self.max_len, rng)
                    yield {"input_ids": ids, "special_tokens_mask": spec}
            epoch += 1


class MLMCollator:
    """BERT MLM masking faithful to the spec: 15% selected; 80% -> [MASK],
    10% -> random canonical nucleotide, 10% unchanged; loss only on selected.
    Dynamic right-padding to a multiple of pad_to (tensor-core friendly)."""

    def __init__(self, mlm_probability=0.15, pad_to=8, seed=42):
        self.p = mlm_probability
        self.pad_to = pad_to
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, features):
        lens = [len(f["input_ids"]) for f in features]
        maxL = max(lens)
        if self.pad_to:
            maxL = ((maxL + self.pad_to - 1) // self.pad_to) * self.pad_to
        B = len(features)
        input_ids = torch.full((B, maxL), PAD, dtype=torch.long)
        attn = torch.zeros((B, maxL), dtype=torch.long)
        special = torch.ones((B, maxL), dtype=torch.bool)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            attn[i, :n] = 1
            special[i, :n] = torch.tensor(f["special_tokens_mask"], dtype=torch.bool)

        labels = input_ids.clone()
        prob = torch.full(input_ids.shape, self.p)
        prob[special] = 0.0
        masked = torch.bernoulli(prob, generator=self.g).bool()
        labels[~masked] = -100                                   # loss only on masked

        # 80% -> [MASK]
        m80 = torch.bernoulli(torch.full(input_ids.shape, 0.8), generator=self.g).bool() & masked
        input_ids[m80] = MASK
        # 10% -> random canonical nucleotide (of the remaining 20%)
        m10 = torch.bernoulli(torch.full(input_ids.shape, 0.5), generator=self.g).bool() & masked & ~m80
        rand = torch.tensor(CANON_IDS)[torch.randint(0, len(CANON_IDS), (int(m10.sum()),), generator=self.g)]
        input_ids[m10] = rand
        # remaining 10% kept unchanged
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the /dev/shm RNAcentral stream cache.")
    ap.add_argument("--gz", default="data/raw/rnacentral_active.fasta.gz")
    ap.add_argument("--shm", default="/dev/shm/rna_seqs.txt")
    ap.add_argument("--val", default="outputs/fm_hybrid_650m/val_seqs.txt")
    ap.add_argument("--val_mod", type=int, default=2048)
    ap.add_argument("--min_len", type=int, default=1)
    a = ap.parse_args()
    n = build_shm_cache(a.gz, a.shm, a.val, a.val_mod, a.min_len)
    print(f"done: {n:,} train seqs cached at {a.shm}")
