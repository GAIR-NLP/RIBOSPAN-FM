# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""CT, dot-bracket, and bpseq readers used by the LIMBER secondary-structure splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CLOSE_TO_OPEN = {")": "(", "]": "[", "}": "{", ">": "<"}
_PARSERS = {".ct": "ct", ".st": "st", ".bpseq": "bpseq"}


def normalize_sequence(sequence: str) -> str:
    sequence = sequence.upper().replace("T", "U")
    return "".join(base if base in "ACGU" else "N" for base in sequence)


def _payload_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not line.lstrip().startswith("#")]


def dot_bracket_to_matrix(dot_bracket: str) -> np.ndarray:
    length = len(dot_bracket)
    matrix = np.zeros((length, length), dtype=np.float32)
    stacks = {symbol: [] for symbol in "([{<"}
    for index, symbol in enumerate(dot_bracket):
        if symbol in stacks:
            stacks[symbol].append(index)
        elif symbol in _CLOSE_TO_OPEN:
            partner = stacks[_CLOSE_TO_OPEN[symbol]].pop()
            matrix[index, partner] = 1.0
        elif symbol != ".":
            raise ValueError(f"unexpected dot-bracket symbol {symbol!r} at {index}")
    return np.minimum(matrix + matrix.T, 1.0)


def parse_st(path: Path) -> tuple[str, np.ndarray]:
    lines = _payload_lines(path)
    sequence, dot_bracket = lines[0].rstrip(), lines[1].rstrip()
    if len(sequence) != len(dot_bracket):
        raise ValueError(f"{path}: sequence length {len(sequence)} != structure length {len(dot_bracket)}")
    return normalize_sequence(sequence), dot_bracket_to_matrix(dot_bracket)


def parse_bpseq(path: Path) -> tuple[str, np.ndarray]:
    lines = [line for line in _payload_lines(path) if line.strip()]
    length = len(lines)
    sequence: list[str] = []
    matrix = np.zeros((length, length), dtype=np.float32)
    for line in lines:
        index_text, base, partner_text = line.split()[:3]
        sequence.append(base)
        index, partner = int(index_text), int(partner_text)
        if partner:
            matrix[index - 1, partner - 1] = 1.0
    return normalize_sequence("".join(sequence)), np.minimum(matrix + matrix.T, 1.0)


def parse_ct(path: Path) -> tuple[str, np.ndarray]:
    lines = [line for line in _payload_lines(path) if line.strip()]
    length = int(lines[0].split()[0])
    matrix = np.zeros((length, length), dtype=np.float32)
    sequence: list[str] = []
    for line in lines[1:1 + length]:
        parts = line.split()
        index, base, partner = int(parts[0]), parts[1], int(parts[4])
        sequence.append(base)
        if partner > 0:
            matrix[index - 1, partner - 1] = 1.0
    return normalize_sequence("".join(sequence)), np.minimum(matrix + matrix.T, 1.0)


def _parse(path: Path) -> tuple[str, np.ndarray]:
    kind = _PARSERS.get(path.suffix.lower())
    if kind == "ct":
        return parse_ct(path)
    if kind == "st":
        return parse_st(path)
    if kind == "bpseq":
        return parse_bpseq(path)
    raise ValueError(f"unsupported structure file {path}")


def load_ss_dir(data_dir: Path, max_len: int) -> tuple[list[dict], int]:
    items: list[dict] = []
    skipped = 0
    bad = 0
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix.lower() not in _PARSERS or not path.is_file():
            continue
        try:
            sequence, pairing = _parse(path)
        except Exception:
            bad += 1
            continue
        if 0 < len(sequence) <= max_len:
            items.append({"id": path.stem, "seq": sequence, "pair": pairing})
        else:
            skipped += 1
    if bad:
        print(f"warn {bad} unparsable files in {data_dir}", flush=True)
    return items, skipped
