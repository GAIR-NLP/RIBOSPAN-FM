#!/usr/bin/env python3
"""Make train/val/test splits for supervised contact-map prediction (Part 2).

A. random split: 80/10/10 over all ArchiveII (seed 42).
B. leave-one-family-out (LFO): for each family, test = that family, train/val =
   the rest (90/10). Families too small to serve as a test set are skipped and
   recorded.

Writes data/contact_eval/splits/archiveII_random_{train,val,test}.jsonl and
archiveII_lfo_{family}_{train,val,test}.jsonl, plus a splits_stats.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = PROJECT_ROOT / "data" / "contact_eval" / "splits"
STATS = PROJECT_ROOT / "outputs" / "contact_eval"


def write_jsonl(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/contact_eval/archiveII.jsonl")
    p.add_argument("--min-len", type=int, default=30)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-family-test", type=int, default=10,
                   help="skip LFO for families smaller than this")
    args = p.parse_args()

    SPLITS.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    recs = [json.loads(l) for l in open(args.jsonl)]
    recs = [r for r in recs if args.min_len <= r["length"] <= args.max_len]
    stats = {"total": len(recs), "families": dict(Counter(r["family"] for r in recs))}

    # ---- A. random 80/10/10 ----
    sh = recs[:]
    rng.shuffle(sh)
    n = len(sh)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    rand = {"train": sh[:n_tr], "val": sh[n_tr:n_tr + n_va], "test": sh[n_tr + n_va:]}
    for k, v in rand.items():
        write_jsonl(SPLITS / f"archiveII_random_{k}.jsonl", v)
    stats["random"] = {k: len(v) for k, v in rand.items()}
    print(f"[random] train/val/test = {stats['random']['train']}/"
          f"{stats['random']['val']}/{stats['random']['test']}")

    # ---- B. leave-one-family-out ----
    by_fam = defaultdict(list)
    for r in recs:
        by_fam[r["family"]].append(r)
    stats["lfo"], skipped = {}, []
    for fam, items in sorted(by_fam.items()):
        if len(items) < args.min_family_test:
            skipped.append({"family": fam, "n": len(items)})
            continue
        rest = [r for r in recs if r["family"] != fam]
        rng.shuffle(rest)
        n_va = max(1, int(0.1 * len(rest)))
        tr, va, te = rest[n_va:], rest[:n_va], items
        write_jsonl(SPLITS / f"archiveII_lfo_{fam}_train.jsonl", tr)
        write_jsonl(SPLITS / f"archiveII_lfo_{fam}_val.jsonl", va)
        write_jsonl(SPLITS / f"archiveII_lfo_{fam}_test.jsonl", te)
        stats["lfo"][fam] = {"train": len(tr), "val": len(va), "test": len(te)}
        print(f"[lfo] heldout={fam:12} train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    stats["lfo_skipped"] = skipped
    if skipped:
        print(f"[lfo] skipped small families: {[s['family'] for s in skipped]}")

    json.dump(stats, open(STATS / "splits_stats.json", "w"), indent=2)
    print(f"[stats] wrote {STATS / 'splits_stats.json'}")


if __name__ == "__main__":
    main()
