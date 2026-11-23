#!/usr/bin/env python3
"""Convert archiveII.jsonl to CT files with leave-one-family-out (LFO) splits.

Output: data/contact_eval/raw/ct/fam-fold/{family}/{train,valid,test}/*.ct

CT format (1-indexed):
  Line 1: <N> <id>
  Lines 2..N+1: <idx> <nuc> <prev> <next> <pair> <idx>
  pair=0 means unpaired; pairs in JSONL are 0-indexed → add 1 for CT.
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL = PROJECT_ROOT / "data" / "contact_eval" / "archiveII.jsonl"
OUT_ROOT = PROJECT_ROOT / "data" / "contact_eval" / "raw" / "ct" / "fam-fold"


def to_ct(record: dict) -> str:
    seq = record["sequence"].upper().replace("T", "U")
    n = len(seq)
    # pairs are 0-indexed in the JSONL
    pair_map = {}
    for a, b in record.get("pairs", []):
        pair_map[a] = b
        pair_map[b] = a
    lines = [f"{n} {record['id']}"]
    for i, nuc in enumerate(seq):
        prev_i = i      # 1-indexed prev = i (0-indexed i maps to 1-indexed i)
        next_i = i + 2  # 1-indexed next = i+2 (0-indexed i+1 maps to 1-indexed i+2)
        j = pair_map.get(i, -1)
        pair_ct = (j + 1) if j >= 0 else 0
        # standard CT: <1idx> <nuc> <prev_1idx> <next_1idx> <pair_1idx> <1idx>
        lines.append(f"{i+1} {nuc} {prev_i} {next_i} {pair_ct} {i+1}")
    return "\n".join(lines) + "\n"


def main():
    recs = [json.loads(l) for l in open(JSONL)]
    by_family = defaultdict(list)
    for r in recs:
        by_family[r["family"]].append(r)

    families = sorted(by_family)
    print(f"Families ({len(families)}): {families}")
    print(f"Total records: {len(recs)}")

    rng = random.Random(42)

    for test_fam in families:
        test_recs = by_family[test_fam]
        other_recs = []
        for fam in families:
            if fam != test_fam:
                other_recs.extend(by_family[fam])

        # 90/10 train/val split on non-test families
        other_shuf = other_recs[:]
        rng.shuffle(other_shuf)
        n_val = max(1, int(len(other_shuf) * 0.1))
        val_recs = other_shuf[:n_val]
        train_recs = other_shuf[n_val:]

        for split, recs_split in [("train", train_recs), ("valid", val_recs), ("test", test_recs)]:
            d = OUT_ROOT / test_fam / split
            d.mkdir(parents=True, exist_ok=True)
            for r in recs_split:
                ct_text = to_ct(r)
                (d / f"{r['id']}.ct").write_text(ct_text)

        print(f"  [{test_fam}] test={len(test_recs)} train={len(train_recs)} val={len(val_recs)}")

    print(f"\nDone → {OUT_ROOT}")


if __name__ == "__main__":
    main()
