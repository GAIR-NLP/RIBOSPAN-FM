# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Experiment YAML, dataset groups, output paths, and frozen tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def public_path(path: Path | str, root: Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def resolve_config_path(path: Path | str | None = None) -> Path:
    value = Path(path) if path is not None else Path("configs/experiment.yaml")
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def load_yaml(path: Path | str) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def load_experiment_config(path: Path | str | None = None) -> tuple[dict[str, Any], Path]:
    config_path = resolve_config_path(path)
    cfg = load_yaml(config_path)
    if cfg.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    raw_root = Path(cfg.get("project_root", "."))
    project_root = (
        raw_root.resolve()
        if raw_root.is_absolute()
        else (PROJECT_ROOT / raw_root).resolve()
    )
    return cfg, project_root


def load_dataset_registry(cfg: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    datasets = dict(cfg.get("datasets") or {})
    data_cfg = cfg.get("data") or {}
    names = list(data_cfg.get("names") or datasets.keys())
    groups: dict[str, dict[str, Any]] = {"all": {"datasets": names}}
    extra = data_cfg.get("groups") or {}
    for key, value in extra.items():
        if isinstance(value, dict) and "datasets" in value:
            groups[str(key)] = value
        elif isinstance(value, list):
            groups[str(key)] = {"datasets": list(value)}
        else:
            raise ValueError(f"data.groups.{key} must be a list of dataset ids")
    return {
        "default_group": str(data_cfg.get("default_group") or "all"),
        "groups": groups,
        "datasets": datasets,
    }


def resolve_dataset_ids(
    registry: Mapping[str, Any],
    *,
    group: str | None = None,
    datasets: list[str] | None = None,
) -> list[str]:
    if datasets:
        unknown = [name for name in datasets if name not in registry.get("datasets", {})]
        if unknown:
            raise KeyError(f"unknown dataset id(s): {unknown}")
        return list(datasets)
    groups = registry.get("groups", {})
    chosen = group or str(registry.get("default_group", "all"))
    if chosen not in groups:
        raise KeyError(f"unknown dataset group {chosen!r}; options={sorted(groups)}")
    ids = list(groups[chosen]["datasets"])
    unknown = [name for name in ids if name not in registry.get("datasets", {})]
    if unknown:
        raise KeyError(f"group {chosen} references unknown dataset id(s): {unknown}")
    return ids


def output_root(cfg: Mapping[str, Any], project_root: Path) -> Path:
    return (project_root / cfg.get("output_root", "outputs/rnagym_fitness")).resolve()


def table_dir(cfg: Mapping[str, Any], project_root: Path, dataset_id: str) -> Path:
    """Frozen official substitution tables."""

    data_cfg = cfg.get("data") or {}
    root = data_cfg.get("root") or cfg.get("data_root", "data")
    return (project_root / root / dataset_id).resolve()


def _read_table(directory: Path, stem: str):
    import pandas as pd

    parquet_path = directory / f"{stem}.parquet"
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"missing {stem}.parquet under {directory}")


def load_tables(directory: Path) -> tuple[Any, dict[str, Any]]:
    """Read ``mutants.parquet`` and ``metadata.json``."""

    import json

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return _read_table(directory, "mutants"), metadata


def results_dir(cfg: Mapping[str, Any], project_root: Path) -> Path:
    return output_root(cfg, project_root) / str(cfg.get("results_subdir", "results"))
