# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collect probe metrics and family-level Z-scores.

Z-score protocol:

1. For each (seed, dataset, sub-task), take the primary metric of every model.
2. Pearson ``r`` is Fisher-transformed (``artanh``) first; AUPRC is left as-is.
3. Z-score **across models** (population std, ``ddof=0``).
4. Mean-aggregate sub-tasks, then sub-datasets (eCLIP: 20 RBP targets, then
   K562/HepG2; GO: MF/BP/CC; VEP: Mendelian/complex; HL/TE: human+mouse).
5. Mean those family Z values across the 10 seeds.
6. Overall Z is the unweighted mean of family Z.

Absolute Z depends on the model pool in ``configs/experiment.yaml``. Ranking
among these models is the quantity we report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (  # noqa: E402
    family_order,
    load_dataset_registry,
    load_experiment_config,
    results_dir,
)


FISHER_CLIP = 0.999999
OVERALL_LABELS = ("overall",)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _primary_metric(task: str, eval_split: str) -> str:
    if task == "regression":
        return f"{eval_split}_r"
    return f"{eval_split}_auprc"


def _target_mean(payload: dict[str, Any], metric_key: str) -> float | None:
    targets = payload.get("targets") or {}
    values: list[float] = []
    for target in targets.values():
        summary = (target or {}).get("summary") or {}
        stats = summary.get(metric_key) or {}
        mean = stats.get("mean")
        if mean is not None and mean == mean:
            values.append(float(mean))
    if not values:
        return None
    return float(sum(values) / len(values))


def fisher_z(r: np.ndarray | float, clip: float = FISHER_CLIP) -> np.ndarray | float:
    """Fisher transform ``artanh(r)`` with clipping so ``|r|=1`` stays finite."""

    scalar = not isinstance(r, np.ndarray)
    values = np.asarray(r, dtype=np.float64)
    clipped = np.clip(values, -clip, clip)
    out = np.arctanh(clipped)
    if scalar:
        return float(out)
    return out


def zscore_across_models(values: np.ndarray, *, ddof: int = 0) -> np.ndarray:
    """``(x - mean) / std`` across the model axis. Constant input → all zeros."""

    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if finite.sum() < 2:
        out = np.full_like(x, np.nan, dtype=np.float64)
        out[finite] = 0.0
        return out
    mu = float(np.mean(x[finite]))
    sd = float(np.std(x[finite], ddof=ddof))
    out = np.full_like(x, np.nan, dtype=np.float64)
    if sd == 0.0 or not math.isfinite(sd):
        out[finite] = 0.0
        return out
    out[finite] = (x[finite] - mu) / sd
    return out


def collect_atomic_rows(
    out_root: Path,
    ds_registry: dict[str, Any],
    *,
    eval_split: str,
) -> pd.DataFrame:
    """One row per (model, seed, dataset, target)."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_root.glob("*/*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset_id = str(payload.get("dataset_id") or metrics_path.parent.parent.name)
        model_name = str(payload.get("model") or metrics_path.parent.name)
        spec = ds_registry.get("datasets", {}).get(dataset_id, {})
        family = spec.get("family")
        for target_name, target in (payload.get("targets") or {}).items():
            task = str(target.get("task") or spec.get("task") or "regression")
            metric_key = _primary_metric(task, eval_split)
            per_seed = target.get("per_seed") or {}
            for seed, metrics in per_seed.items():
                raw = metrics.get(metric_key)
                if raw is None:
                    continue
                raw_f = float(raw)
                score = float(fisher_z(raw_f)) if task == "regression" else raw_f
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "family": family,
                        "model": model_name,
                        "target_col": target.get("target_col") or target_name,
                        "task": task,
                        "metric": metric_key,
                        "seed": str(seed),
                        "raw": raw_f,
                        "score": score,
                    }
                )
    return pd.DataFrame(rows)


def attach_subtask_z(atomic: pd.DataFrame) -> pd.DataFrame:
    """Z-score across models inside each (seed, dataset, sub-task)."""

    if atomic.empty:
        return atomic.assign(z=pd.Series(dtype=np.float64))
    frame = atomic.copy()
    z_values = np.full(len(frame), np.nan, dtype=np.float64)
    grouped = frame.groupby(["seed", "dataset_id", "target_col"], sort=False)
    for _, index in grouped.groups.items():
        idx = np.asarray(list(index), dtype=int)
        z_values[idx] = zscore_across_models(frame.loc[idx, "score"].to_numpy())
    frame["z"] = z_values
    return frame


def _mean_ci(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "ci95": math.nan, "n": 0}
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    se = std / math.sqrt(finite.size)
    return {"mean": mean, "std": std, "ci95": 1.96 * se, "n": int(finite.size)}


def aggregate_hierarchy(atomic_z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sub-tasks → sub-datasets → family, still per (model, seed)."""

    if atomic_z.empty:
        empty = pd.DataFrame(columns=["model", "seed", "family", "z"])
        return empty, empty
    dataset_seed = (
        atomic_z.groupby(["model", "seed", "dataset_id", "family"], as_index=False)["z"]
        .mean()
    )
    family_seed = (
        dataset_seed.groupby(["model", "seed", "family"], as_index=False)["z"]
        .mean()
    )
    return dataset_seed, family_seed


def summarize_families(family_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, model), part in family_seed.groupby(["family", "model"]):
        stats = _mean_ci(part["z"].to_numpy())
        rows.append({"family": family, "model": model, **stats})
    return pd.DataFrame(rows)


def overall_by_seed(
    family_seed: pd.DataFrame,
    families: tuple[str, ...],
    *,
    label: str,
) -> pd.DataFrame:
    subset = family_seed[family_seed["family"].isin(families)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["label", "model", "seed", "z", "n_families"])
    n_families = int(subset["family"].nunique())
    counts = subset.groupby(["model", "seed"])["family"].nunique()
    keep = counts[counts == n_families].index
    subset = subset.set_index(["model", "seed"]).loc[keep].reset_index()
    overall = (
        subset.groupby(["model", "seed"], as_index=False)["z"]
        .mean()
        .assign(label=label, n_families=n_families)
    )
    return overall[["label", "model", "seed", "z", "n_families"]]


def summarize_overall(overall_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (label, model), part in overall_seed.groupby(["label", "model"]):
        stats = _mean_ci(part["z"].to_numpy())
        rows.append(
            {
                "label": label,
                "model": model,
                "n_families": int(part["n_families"].iloc[0]) if len(part) else 0,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def compute_zscore_tables(
    out_root: Path,
    ds_registry: dict[str, Any],
    *,
    eval_split: str,
    model_order: list[str] | None = None,
) -> dict[str, Any]:
    atomic = collect_atomic_rows(out_root, ds_registry, eval_split=eval_split)
    atomic_z = attach_subtask_z(atomic)
    dataset_seed, family_seed = aggregate_hierarchy(atomic_z)
    by_family = summarize_families(family_seed)
    families = family_order(ds_registry)
    overall_seed = overall_by_seed(family_seed, families, label="overall")
    overall = summarize_overall(overall_seed)

    if model_order:
        family_cat = pd.CategoricalDtype(list(families), ordered=True)
        model_cat = pd.CategoricalDtype(model_order, ordered=True)
        if not by_family.empty:
            by_family["family"] = by_family["family"].astype(family_cat)
            by_family["model"] = by_family["model"].astype(model_cat)
            by_family = by_family.sort_values(["family", "model"]).reset_index(drop=True)
        if not overall.empty:
            overall["label"] = pd.Categorical(
                overall["label"], categories=list(OVERALL_LABELS), ordered=True
            )
            overall["model"] = overall["model"].astype(model_cat)
            overall = overall.sort_values(["label", "model"]).reset_index(drop=True)

    n_models = int(atomic["model"].nunique()) if not atomic.empty else 0
    n_seeds = int(atomic["seed"].nunique()) if not atomic.empty else 0
    families_present = (
        sorted({str(v) for v in family_seed["family"].dropna().unique()})
        if not family_seed.empty
        else []
    )
    return {
        "eval_split": eval_split,
        "n_models": n_models,
        "n_seeds": n_seeds,
        "model_pool": sorted(atomic["model"].unique().tolist()) if not atomic.empty else [],
        "families_present": families_present,
        "overall_families": list(families),
        "note": (
            "Z is computed across the models in this run. Ranking among these "
            "models is the reportable quantity."
        ),
        "by_family": by_family,
        "overall": overall,
        "family_seed": family_seed,
        "atomic": atomic_z,
    }


def write_zscore_tables(
    out_root: Path,
    ds_registry: dict[str, Any],
    *,
    eval_split: str,
    model_order: list[str] | None = None,
) -> dict[str, Any]:
    payload = compute_zscore_tables(
        out_root, ds_registry, eval_split=eval_split, model_order=model_order
    )
    by_family: pd.DataFrame = payload["by_family"]
    overall: pd.DataFrame = payload["overall"]
    out_root.mkdir(parents=True, exist_ok=True)
    family_path = out_root / "zscore_by_family.tsv"
    overall_path = out_root / "zscore_overall.tsv"
    json_path = out_root / "zscore.json"
    if by_family.empty:
        family_path.write_text("family\tmodel\tmean\tstd\tci95\tn\n", encoding="utf-8")
    else:
        by_family.to_csv(family_path, sep="\t", index=False)
    if overall.empty:
        overall_path.write_text("label\tmodel\tn_families\tmean\tstd\tci95\tn\n", encoding="utf-8")
    else:
        overall.to_csv(overall_path, sep="\t", index=False)
    serializable = {
        key: value
        for key, value in payload.items()
        if key not in {"by_family", "overall", "family_seed", "atomic"}
    }
    serializable["by_family"] = by_family.to_dict(orient="records") if not by_family.empty else []
    serializable["overall"] = overall.to_dict(orient="records") if not overall.empty else []
    _write_json(json_path, serializable)
    payload["paths"] = {
        "by_family": str(family_path),
        "overall": str(overall_path),
        "json": str(json_path),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--eval-split", choices=("val", "test"), default=None)
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    ds_registry = load_dataset_registry(cfg, project_root)
    eval_split = args.eval_split or str(cfg.get("probe", {}).get("eval_split", "val"))
    out_root = results_dir(cfg, project_root)

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_root.glob("*/*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset_id = payload.get("dataset_id") or metrics_path.parent.parent.name
        model_name = payload.get("model") or metrics_path.parent.name
        spec = ds_registry.get("datasets", {}).get(dataset_id, {})
        task = str(spec.get("task") or "regression")
        if payload.get("targets"):
            task = next(iter(payload["targets"].values())).get("task") or task
        metric_key = _primary_metric(str(task), eval_split)
        mean = _target_mean(payload, metric_key)
        rows.append(
            {
                "dataset_id": dataset_id,
                "family": spec.get("family"),
                "model": model_name,
                "task": task,
                "metric": metric_key,
                "score": mean,
                "metrics_path": str(metrics_path.relative_to(project_root)),
            }
        )

    frame = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    tsv_path = out_root / "summary.tsv"
    json_path = out_root / "summary.json"
    grouped_path = out_root / "summary_by_family.tsv"
    if frame.empty:
        print(f"no metrics under {out_root}", flush=True)
        _write_json(json_path, [])
        tsv_path.write_text("dataset_id\tfamily\tmodel\tscore\n", encoding="utf-8")
        grouped_path.write_text("family\tmodel\tscore\n", encoding="utf-8")
        return 0
    frame.sort_values(["family", "dataset_id", "model"]).to_csv(tsv_path, sep="\t", index=False)
    _write_json(json_path, frame.to_dict(orient="records"))

    grouped = (
        frame.dropna(subset=["score"])
        .groupby(["family", "model"], as_index=False)["score"]
        .mean()
    )
    grouped.to_csv(grouped_path, sep="\t", index=False)
    print(f"wrote {tsv_path}", flush=True)
    print(f"wrote {grouped_path}", flush=True)
    if not grouped.empty:
        print(grouped.to_string(index=False), flush=True)
    model_order = list(cfg.get("models", {}).get("names") or [])
    zscore = write_zscore_tables(
        out_root, ds_registry, eval_split=eval_split, model_order=model_order
    )
    print(f"wrote {zscore['paths']['by_family']}", flush=True)
    print(f"wrote {zscore['paths']['overall']}", flush=True)
    overall = zscore["overall"]
    if not overall.empty:
        print("\nZ-score overall", flush=True)
        print(overall.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
