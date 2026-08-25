# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Experiment YAML, sequence-table I/O, and display-name aliases."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def public_path(path: Path | str, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible."""

    value = Path(path)
    try:
        return value.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Resolve ``--config`` relative to the repository root, not the CWD."""

    value = Path(path) if path is not None else Path("configs/experiment.yaml")
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def load_experiment_config(path: Path | str | None = None) -> tuple[dict[str, Any], Path]:
    """Load experiment YAML; relative ``project_root`` is repo-root relative."""

    import yaml

    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    raw_root = Path(cfg.get("project_root", "."))
    project_root = (
        raw_root.resolve()
        if raw_root.is_absolute()
        else (PROJECT_ROOT / raw_root).resolve()
    )
    return cfg, project_root


def resolve_dataset_path(cfg: Mapping[str, Any], project_root: Path) -> Path:
    data_cfg = cfg["data"]
    return (project_root / data_cfg["path"]).resolve()


def load_sequence_table(
    path: Path | str,
    *,
    id_column: str = "id",
    type_column: str = "rna_type",
    sequence_column: str = "sequence",
    header_column: str = "header",
) -> list[dict[str, Any]]:
    """Read the frozen TSV; rows use ``seq_id`` / ``rna_type`` / ``sequence``."""

    table = Path(path)
    if not table.is_file():
        raise FileNotFoundError(f"sequence table not found: {table}")
    rows: list[dict[str, Any]] = []
    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{table} has no header")
        missing = {id_column, type_column, sequence_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{table} is missing columns {sorted(missing)}")
        has_header = header_column in reader.fieldnames
        has_truncated = "store_truncated" in reader.fieldnames
        has_length = "length" in reader.fieldnames
        for index, raw in enumerate(reader, start=2):
            seq_id = (raw.get(id_column) or "").strip()
            sequence = (raw.get(sequence_column) or "").strip()
            rna_type = (raw.get(type_column) or "").strip() or "UNK"
            if not seq_id:
                raise ValueError(f"{table} row {index} has an empty id")
            if not sequence:
                raise ValueError(f"{table} row {index} has an empty sequence")
            length = int(raw["length"]) if has_length and raw.get("length") else len(sequence)
            truncated = False
            if has_truncated:
                truncated = str(raw.get("store_truncated") or "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                }
            rows.append(
                {
                    "seq_id": seq_id,
                    "rna_type": rna_type,
                    "sequence": sequence,
                    "length": length,
                    "header": (raw.get(header_column) or seq_id) if has_header else seq_id,
                    "store_truncated": truncated,
                }
            )
    if not rows:
        raise ValueError(f"{table} contains no sequences")
    return rows


DISPLAY_NAMES = {
    "AIDO.RNA-1.6B": "AIDO.RNA",
    "AIDO.RNA-1.6B-CDS": "AIDO.RNA-CDS",
    "RIBOSCOPE-1.6B-run4": "RIBOSPAN-1K-15",
    "RIBOSCOPE-1.6B-run4-2": "RIBOSPAN-1K-40",
    "RIBOSCOPE-1.6B-run5": "RIBOSPAN-10K-15",
    "RIBOSCOPE-1.6B-run5-2": "RIBOSPAN-10K-40",
    "RiNALMo-giga": "RiNALMo",
}


def display_model_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)

