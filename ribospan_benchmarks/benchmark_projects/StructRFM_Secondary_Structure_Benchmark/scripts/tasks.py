# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""StructRFM secondary-structure splits shipped in tasks_data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SSP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "structrfm_release"
    / "tasks_data"
    / "seqcls_ssp"
    / "data"
    / "ssp"
)
# structRFM's collator keeps at most 511 nucleotides so that CLS plus bases fit in 512.
MAX_NUCLEOTIDES = 511

TASKS = {
    "bpRNA1m": {
        "train": "TR0",
        "test": "TS0",
        "test_max_length": None,
    },
    "RNAStrAlign": {
        "train": "RNAStrAlign_bpseq",
        "test": "archiveII_bpseq",
        "test_max_length": 599,
    },
}


def list_bpseq(directory: Path, *, max_length: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in {".bpseq", ".ct"} and path.is_file()
    )
    if max_length is None:
        return paths
    kept: list[Path] = []
    for path in paths:
        seq, _pairs = load_bpseq(path)
        if len(seq) <= max_length:
            kept.append(path)
    return kept


def load_bpseq(path: Path) -> tuple[str, np.ndarray]:
    pairs = [0]
    bases: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        base = parts[1]
        partner = int(parts[2])
        bases.append(base)
        pairs.append(partner)
    return "".join(bases), np.asarray(pairs, dtype=np.int32)


def prepare_example(path: Path) -> tuple[str, np.ndarray]:
    """Match structRFM: U alphabet, truncate to 511 nt, drop dangling pairs."""

    seq, pairs = load_bpseq(path)
    seq = seq.upper().replace("T", "U")
    if len(seq) > MAX_NUCLEOTIDES:
        seq = seq[:MAX_NUCLEOTIDES]
        pairs = pairs[: MAX_NUCLEOTIDES + 1].copy()
        pairs[pairs >= len(pairs)] = 0
    return seq, pairs
