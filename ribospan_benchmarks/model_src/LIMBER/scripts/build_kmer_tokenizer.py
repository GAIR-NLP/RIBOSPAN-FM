#!/usr/bin/env python3
"""Build a non-overlapping k-mer BERT vocab.txt for an arbitrary k.

vocab = 5 special tokens + all 5^k k-mers over {A, U, C, G, N}
=> size 5^k + 5. Loadable by BertTokenizer / BertTokenizerFast.

Run:
    python scripts/build_kmer_tokenizer.py --k 4
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
ALPHABET = ["A", "U", "C", "G", "N"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    kmers = ["".join(t) for t in product(ALPHABET, repeat=args.k)]
    tokens = SPECIAL_TOKENS + kmers
    out_dir = args.output_dir or (PROJECT_ROOT / "tokenizers" / f"kmer{args.k}")
    out_path = out_dir / "vocab.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for tok in tokens:
            fh.write(tok + "\n")
    assert len(tokens) == 5 ** args.k + 5
    print(f"[tokenizer] kmer{args.k} vocab: {len(tokens)} tokens -> {out_path}")


if __name__ == "__main__":
    main()
