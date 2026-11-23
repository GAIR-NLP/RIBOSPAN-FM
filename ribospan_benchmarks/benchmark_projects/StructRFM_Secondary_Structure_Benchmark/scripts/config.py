# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the holding-tree registry. No encoder is loaded."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def registry_rows(config_path: Path | None = None) -> list[dict[str, Any]]:
    config_path = config_path or (PROJECT_ROOT / "configs" / "experiment.yaml")
    experiment = load_yaml(config_path)
    registry_path = Path(experiment["models_registry"])
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    document = load_yaml(registry_path)
    models_root = Path(document.get("models_root", "../model_weights"))
    if not models_root.is_absolute():
        models_root = (registry_path.parent / models_root).resolve()
    rows: list[dict[str, Any]] = []
    for entry in document["models"]:
        path = models_root / str(entry["source_path"])
        rows.append(
            {
                "name": str(entry["name"]),
                "backend": str(entry["backend"]),
                "context_window": int(entry["context_window"]),
                "path": path,
                "exists": path.exists(),
                "checkpoints": _checkpoint_status(str(entry["name"]), path),
            }
        )
    return rows


def _checkpoint_status(name: str, path: Path) -> list[tuple[str, bool, int]]:
    """Return ``(relative path, complete, bytes)`` for the files this entry needs."""

    expected: list[tuple[str, int | None]]
    if name == "structRFM":
        expected = [
            ("structRFM_checkpoint/model.safetensors", 344231992),
            ("structRFM_SSP_bpRNA1m.pt", 348179698),
            ("structRFM_SSP_RNAStrAlign.pt", 348179375),
        ]
    elif name.startswith("RIBOSPAN-"):
        expected = [("config.json", None), ("pytorch_model.bin", None)]
    else:
        expected = []
    found: list[tuple[str, bool, int]] = []
    for relative, size in expected:
        file_path = path / relative
        nbytes = file_path.stat().st_size if file_path.is_file() else 0
        if file_path.is_dir():
            nbytes = sum(item.stat().st_size for item in file_path.rglob("*") if item.is_file())
            complete = nbytes > 0
        elif size is None:
            complete = file_path.is_file() and nbytes > 0
        else:
            complete = nbytes == size
        found.append((relative, complete, nbytes))
    return found
