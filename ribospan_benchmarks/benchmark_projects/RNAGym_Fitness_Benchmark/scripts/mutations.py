# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parse RNAGym mutant strings (``A123C`` or comma-separated multi-mutants)."""

from __future__ import annotations

import re
from dataclasses import dataclass


SINGLE_MUT = re.compile(r"^([ACGTU])([0-9]+)([ACGTU])$")


@dataclass(frozen=True)
class PointMutation:
    wt: str
    position: int  # 0-based
    mt: str

    def as_tuple(self) -> tuple[str, int, str]:
        return self.wt, self.position, self.mt


def normalize_base(base: str) -> str:
    value = str(base).strip().upper().replace("U", "T")
    if value not in {"A", "C", "G", "T"}:
        raise ValueError(f"unsupported base {base!r}")
    return value


def parse_mutant(raw: str) -> list[PointMutation] | None:
    """Return substitutions, or ``None`` for indels / malformed rows."""

    text = str(raw).strip().upper().replace(" ", "").replace("T", "T").replace("U", "U")
    if not text or text.lower() in {"nan", "none", "wt", "wildtype", "wild-type"}:
        return None
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    parsed: list[PointMutation] = []
    for part in parts:
        token = part.replace("U", "T")
        match = SINGLE_MUT.match(token)
        if match is None:
            return None
        wt, pos_s, mt = match.group(1), match.group(2), match.group(3)
        wt_n = normalize_base(wt)
        mt_n = normalize_base(mt)
        if wt_n == mt_n:
            return None
        parsed.append(PointMutation(wt=wt_n, position=int(pos_s) - 1, mt=mt_n))
    positions = [item.position for item in parsed]
    if len(positions) != len(set(positions)):
        return None
    return parsed


def apply_mutations(sequence: str, mutations: list[PointMutation]) -> str:
    chars = list(sequence)
    for item in mutations:
        if item.position < 0 or item.position >= len(chars):
            raise IndexError(f"position {item.position} out of range for length {len(chars)}")
        if chars[item.position] != item.wt:
            raise ValueError(
                f"WT mismatch at {item.position}: sequence={chars[item.position]} expected={item.wt}"
            )
        chars[item.position] = item.mt
    return "".join(chars)


def mutation_window(
    length: int,
    positions: list[int],
    *,
    max_nucleotides: int,
) -> tuple[int, int]:
    """Mutation-centered crop used by RNAGym RNA-FM when the WT exceeds the window."""

    if max_nucleotides < 1:
        raise ValueError("max_nucleotides must be positive")
    if length <= max_nucleotides:
        return 0, length
    if not positions:
        return 0, max_nucleotides
    min_idx, max_idx = min(positions), max(positions)
    if max_idx - min_idx + 1 > max_nucleotides:
        raise ValueError(
            f"mutations span {max_idx - min_idx + 1} nt, longer than window {max_nucleotides}"
        )
    center = (min_idx + max_idx) // 2
    half = max_nucleotides // 2
    start = max(0, center - half)
    end = min(length, start + max_nucleotides)
    if end == length:
        start = max(0, length - max_nucleotides)
        end = length
    if not all(start <= idx < end for idx in positions):
        raise ValueError(f"window [{start}, {end}) misses mutations {positions}")
    return start, end
