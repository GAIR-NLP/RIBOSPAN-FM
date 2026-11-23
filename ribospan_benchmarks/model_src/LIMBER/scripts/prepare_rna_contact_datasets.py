#!/usr/bin/env python3
"""Prepare RNA secondary-structure datasets for unsupervised contact probing.

Supports ArchiveII (CT files, family splits from marcellszi/dl-rna, referenced
by RiNALMo) and a bpRNA subset (.dbn files). Parses structure to a base-pair
list (dot-bracket parsing supports pseudoknot brackets () [] {} <> and letter
pairs Aa..Zz), normalizes sequences (T->U, non-AUCGN -> N), filters by length,
caps per family, and writes unified JSONL + stats.

If local data is absent and --download is given, ArchiveII (and optionally
bpRNA) are fetched from the URLs in DOWNLOAD_URLS. You may also point at local
files/dirs via --archiveii-ct-dir / --bprna-dbn-dir.

Output (one JSON object per line):
  {"id","family","sequence","dot_bracket","pairs":[[i,j],...],"length","source"}

Run:
  python scripts/prepare_rna_contact_datasets.py --datasets archiveII \
      --archiveii-ct-dir data/contact_eval/raw/ct/fam-fold
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "contact_eval" / "raw"
OUT_DATA = PROJECT_ROOT / "data" / "contact_eval"
OUT_STATS = PROJECT_ROOT / "outputs" / "contact_eval"

# Reserved download interface (URLs from RiNALMo resources/remote_data.json).
DOWNLOAD_URLS = {
    "archiveII": "https://github.com/marcellszi/dl-rna/releases/download/Data/ct-splits.tar.gz",
    "bprna": "https://dl.dropboxusercontent.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip",
}

VALID = set("AUCGN")
# open -> close for dot-bracket generation / parsing (incl. common pseudoknots)
BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
LETTER_PAIRS = [(chr(o), chr(o + 32)) for o in range(ord("A"), ord("Z") + 1)]
OPEN2CLOSE = dict(BRACKETS + LETTER_PAIRS)
CLOSE2OPEN = {c: o for o, c in OPEN2CLOSE.items()}


def normalize(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in VALID else "N" for c in seq)


def parse_ct(path: Path):
    """ArchiveII CT file -> (sequence, pairs[(i,j) 0-based, i<j], title)."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    header = lines[0].split()
    try:
        n = int(header[0])
    except ValueError:
        return None
    title = " ".join(header[1:])
    seq, pairs = [], []
    for ln in lines[1:1 + n]:
        p = ln.split()
        if len(p) < 5:
            continue
        idx, base, partner = int(p[0]), p[1], int(p[4])
        seq.append(base)
        if partner > idx:                       # keep i<j once
            pairs.append((idx - 1, partner - 1))
    return "".join(seq), pairs, title


def parse_dotbracket(db: str, warn=True):
    """Dot-bracket -> pairs[(i,j) 0-based, i<j]. Supports () [] {} <> and Aa..Zz.
    Malformed (unmatched / unknown chars) emits a warning instead of failing
    silently."""
    stacks = defaultdict(list)
    pairs = []
    n_unmatched_close = 0
    for i, ch in enumerate(db):
        if ch in OPEN2CLOSE:
            stacks[ch].append(i)
        elif ch in CLOSE2OPEN:
            o = CLOSE2OPEN[ch]
            if stacks[o]:
                j = stacks[o].pop()
                pairs.append((j, i))
            else:
                n_unmatched_close += 1
        elif ch not in ".":
            if warn:
                print(f"[warn] dot-bracket: unknown symbol {ch!r} at {i}")
    n_unmatched_open = sum(len(v) for v in stacks.values())
    if warn and (n_unmatched_open or n_unmatched_close):
        print(f"[warn] malformed dot-bracket: {n_unmatched_open} unmatched open, "
              f"{n_unmatched_close} unmatched close")
    return sorted(pairs)


def pairs_to_dotbracket(length: int, pairs):
    """Assign pairs to non-crossing bracket levels -> dot-bracket string."""
    levels = []  # each level is a list of (i,j) pairs that don't cross
    db = ["."] * length

    def crosses(a, b):
        (i, j), (k, l) = a, b
        return (i < k < j < l) or (k < i < l < j)

    order = BRACKETS + LETTER_PAIRS
    for (i, j) in sorted(pairs):
        placed = False
        for lvl, plist in enumerate(levels):
            if not any(crosses((i, j), pr) for pr in plist):
                plist.append((i, j))
                o, c = order[min(lvl, len(order) - 1)]
                db[i], db[j] = o, c
                placed = True
                break
        if not placed:
            o, c = order[min(len(levels), len(order) - 1)]
            levels.append([(i, j)])
            db[i], db[j] = o, c
    return "".join(db)


def iter_archiveii(ct_dir: Path):
    """Yield (id, family, seq, pairs) from an ArchiveII fam-fold CT tree.

    fam-fold is a leave-one-family-out CV layout: each fam-fold/{family}/ dir
    contains the WHOLE dataset, with that family held out in test/. So the
    family's own sequences are exactly fam-fold/{family}/test/*.ct — the union
    over families gives the full ArchiveII set, each sequence once, correctly
    labelled. (Falls back to scanning all *.ct if no test/ dirs exist.)"""
    cts = sorted(ct_dir.rglob("test/*.ct")) or sorted(ct_dir.rglob("*.ct"))
    seen = set()
    for ct in cts:
        parsed = parse_ct(ct)
        if not parsed:
            continue
        seq, pairs, _ = parsed
        seq = normalize(seq)
        family = ct.parent.parent.name if ct.parent.name == "test" else "unknown"
        if seq in seen:
            continue
        seen.add(seq)
        yield ct.stem, family, seq, pairs


def iter_bprna(dbn_dir: Path):
    """Yield (id, family, seq, pairs) from bpRNA .dbn files."""
    for dbn in sorted(dbn_dir.rglob("*.dbn")):
        lines = [ln.rstrip() for ln in dbn.read_text().splitlines()]
        name, family = dbn.stem, "bprna"
        body = []
        for ln in lines:
            if ln.startswith("#"):
                low = ln.lower()
                if "name" in low:
                    name = ln.split(":", 1)[-1].strip() or name
                continue
            if ln.strip():
                body.append(ln.strip())
        if len(body) < 2:
            continue
        seq, db = normalize(body[0]), body[1]
        if len(seq) != len(db):
            continue
        yield name, family, seq, parse_dotbracket(db)


def maybe_download(which: str):
    url = DOWNLOAD_URLS[which]
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / Path(url.split("?")[0]).name
    if not dest.exists():
        print(f"[download] {which}: {url}")
        urllib.request.urlretrieve(url, dest)
    if dest.suffixes[-2:] == [".tar", ".gz"] or dest.suffix == ".tgz":
        with tarfile.open(dest) as t:
            t.extractall(RAW)
    return dest


def build(records, source, min_len, max_len, max_per_family, rng):
    by_fam = defaultdict(list)
    for rid, family, seq, pairs in records:
        L = len(seq)
        if L < min_len or L > max_len:
            continue
        pairs = [[i, j] for (i, j) in pairs if 0 <= i < j < L]
        by_fam[family].append({
            "id": rid, "family": family, "sequence": seq,
            "dot_bracket": pairs_to_dotbracket(L, pairs),
            "pairs": pairs, "length": L, "source": source,
        })
    out = []
    for fam, items in by_fam.items():
        rng.shuffle(items)
        out.extend(items[:max_per_family])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["archiveII"],
                   choices=["archiveII", "bprna_subset"])
    p.add_argument("--archiveii-ct-dir", default=str(RAW / "ct" / "fam-fold"))
    p.add_argument("--bprna-dbn-dir", default=str(RAW / "bpRNA" / "dbn"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-len", type=int, default=30)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--max-per-family", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(OUT_DATA))
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    stats = {}

    if "archiveII" in args.datasets:
        ct_dir = Path(args.archiveii_ct_dir)
        if not ct_dir.exists() and args.download:
            maybe_download("archiveII")
        if not ct_dir.exists():
            print(f"[warn] ArchiveII CT dir missing: {ct_dir} (use --download)")
        else:
            recs = build(iter_archiveii(ct_dir), "archiveII",
                         args.min_len, args.max_len, args.max_per_family, rng)
            path = out_dir / "archiveII.jsonl"
            with open(path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            fam = Counter(r["family"] for r in recs)
            stats["archiveII"] = {
                "num_sequences": len(recs), "families": dict(fam),
                "len_min": min((r["length"] for r in recs), default=0),
                "len_max": max((r["length"] for r in recs), default=0),
                "mean_pairs": (sum(len(r["pairs"]) for r in recs) / len(recs)) if recs else 0,
                "path": str(path)}
            print(f"[archiveII] {len(recs)} seqs across {len(fam)} families -> {path}")

    if "bprna_subset" in args.datasets:
        dbn_dir = Path(args.bprna_dbn_dir)
        if not dbn_dir.exists() and args.download:
            try:
                maybe_download("bprna")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] bpRNA download failed: {e}")
        if not dbn_dir.exists():
            print(f"[warn] bpRNA dbn dir missing: {dbn_dir} "
                  f"(provide --bprna-dbn-dir or --download)")
        else:
            recs = build(iter_bprna(dbn_dir), "bprna",
                         args.min_len, args.max_len, args.max_per_family, rng)
            path = out_dir / "bprna_subset.jsonl"
            with open(path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            stats["bprna_subset"] = {"num_sequences": len(recs), "path": str(path)}
            print(f"[bprna_subset] {len(recs)} seqs -> {path}")

    with open(OUT_STATS / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] wrote {OUT_STATS / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
