# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Encoder checkouts under ``ribospan_benchmarks/model_src/``."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC_ROOT = PROJECT_ROOT.parents[1] / "model_src"
RNABERT_ROOT = MODEL_SRC_ROOT / "rnabert"
RIBOSPAN_ROOT = MODEL_SRC_ROOT / "ribospan"
HYDRARNA_SRC_ROOT = MODEL_SRC_ROOT / "HydraRNA"
RINALMO_SRC_ROOT = MODEL_SRC_ROOT / "RiNALMo"
RNAVERSE_SRC_ROOT = MODEL_SRC_ROOT / "RNAverse"
RIBOGRAM_SRC_ROOT = MODEL_SRC_ROOT / "RiboGRAM"


def ensure_import_path() -> Path:
    """Put ``model_src/``, RiNALMo, RNAverse, and RiboGRAM on ``sys.path`` when present."""

    src = str(MODEL_SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    for extra in (RINALMO_SRC_ROOT, RNAVERSE_SRC_ROOT, RIBOGRAM_SRC_ROOT):
        if extra.is_dir():
            extra_s = str(extra)
            if extra_s not in sys.path:
                sys.path.insert(0, extra_s)
    return MODEL_SRC_ROOT
