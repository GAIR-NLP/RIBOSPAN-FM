# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Experiment YAML, model registry, and output paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(path: Path | str | None = None) -> Path:
    value = Path(path) if path is not None else Path("configs/experiment.yaml")
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def load_yaml(path: Path | str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def load_experiment_config(path: Path | str | None = None) -> tuple[dict[str, Any], Path]:
    config_path = resolve_config_path(path)
    config = load_yaml(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    return config, PROJECT_ROOT


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def model_rows(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    registry = _resolve(root, str(config["models_registry"]))
    if registry is None:
        raise ValueError("models_registry is required")
    document = load_yaml(registry)
    rows = []
    for entry in document["models"]:
        weights = _resolve(root, str(entry["weights"]))
        rows.append(
            {
                "name": str(entry["name"]),
                "backend": str(entry["backend"]),
                "weights": weights,
                "context_window": int(entry["context_window"]),
            }
        )
    return rows


def task_spec(config: dict[str, Any], root: Path, name: str) -> dict[str, Any]:
    tasks = config.get("tasks") or {}
    if name not in tasks:
        known = ", ".join(tasks)
        raise KeyError(f"unknown task {name}; known tasks: {known}")
    spec = dict(tasks[name])
    for key in ("train", "valid", "test", "root"):
        if key in spec:
            spec[key] = _resolve(root, spec[key])
    return spec


def output_dir(
    config: dict[str, Any],
    root: Path,
    model: str,
    task: str,
    family: str | None = None,
) -> Path:
    relative = Path(config.get("output_root", "outputs/limber_ss"))
    if not relative.is_absolute():
        relative = root / relative
    path = relative / model / task
    if family:
        path = path / family
    return path
