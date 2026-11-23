#!/usr/bin/env python3
"""Exact-duplicate removal on a FASTA file, keeping the first record per unique
sequence (case-insensitive). Mirrors the RiNALMo preprocessing step that
de-duplicates RNAcentral to ~36M unique ncRNA sequences before mmseqs clustering.

Usage: python dedup_fasta.py <in.fasta> <out.fasta>
"""
import sys
import hashlib

def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    seen = set()
    n_in = 0
    n_out = 0
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        header = None
        seq_chunks = []

        def flush():
            nonlocal n_in, n_out
            if header is None:
                return
            n_in += 1
            seq = "".join(seq_chunks).upper()
            h = hashlib.blake2b(seq.encode(), digest_size=16).digest()
            if h not in seen:
                seen.add(h)
                fout.write(header)
                fout.write("\n")
                for i in range(0, len(seq), 60):
                    fout.write(seq[i:i + 60])
                    fout.write("\n")
                n_out += 1

        for line in fin:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
        flush()

    print(f"[dedup] input records: {n_in}")
    print(f"[dedup] unique records: {n_out}")

if __name__ == "__main__":
    main()
