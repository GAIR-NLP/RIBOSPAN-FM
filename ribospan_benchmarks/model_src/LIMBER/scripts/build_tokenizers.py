#!/usr/bin/env python3
"""Build WordPiece-style vocab.txt files for the two tokenizers we compare.

  - single  : one token per nucleotide (A U C G N) + special tokens
  - center3 : one token per centered 3-mer over alphabet {A U C G N}
              => 5^3 + 5 special = 130 tokens

Both vocab.txt files are loadable by HuggingFace BertTokenizer / BertTokenizerFast.

Run:
    python scripts/build_tokenizers.py
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
ALPHABET = ["A", "U", "C", "G", "N"]


def write_vocab(path: Path, tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for tok in tokens:
            fh.write(tok + "\n")


def build_single() -> list[str]:
    return SPECIAL_TOKENS + ALPHABET


def build_center3() -> list[str]:
    # All 3-mers over {A,U,C,G,N} in deterministic order: AAA, AAU, AAC, ...
    kmers = ["".join(p) for p in product(ALPHABET, repeat=3)]
    return SPECIAL_TOKENS + kmers


def main(argv=None):
    p = argparse.ArgumentParser(description="Build tokenizer vocab.txt files.")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "tokenizers")
    args = p.parse_args(argv)

    single = build_single()
    center3 = build_center3()
    # Non-overlapping 3-mer (stride 3) uses the SAME 3-mer vocabulary as the
    # centered variant; only the data tokenization differs.
    kmer3 = build_center3()

    single_path = args.output_dir / "single" / "vocab.txt"
    center3_path = args.output_dir / "center3" / "vocab.txt"
    kmer3_path = args.output_dir / "kmer3" / "vocab.txt"
    write_vocab(single_path, single)
    write_vocab(center3_path, center3)
    write_vocab(kmer3_path, kmer3)

    print(f"[tokenizer] single  vocab: {len(single)} tokens -> {single_path}")
    print(f"[tokenizer] center3 vocab: {len(center3)} tokens -> {center3_path}")
    print(f"[tokenizer] kmer3   vocab: {len(kmer3)} tokens -> {kmer3_path}")
    assert len(single) == 10, len(single)
    assert len(center3) == 5 ** 3 + 5, len(center3)
    assert len(kmer3) == 5 ** 3 + 5, len(kmer3)


if __name__ == "__main__":
    main()
