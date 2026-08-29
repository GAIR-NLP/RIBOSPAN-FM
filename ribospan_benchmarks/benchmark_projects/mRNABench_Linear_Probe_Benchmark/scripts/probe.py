# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Linear probe on frozen embeddings (RidgeCV / logistic, 10 seeds)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (  # noqa: E402
    embeddings_dir,
    load_dataset_registry,
    load_experiment_config,
    load_tables,
    resolve_dataset_ids,
    results_dir,
    table_dir,
)


def _load_embeddings(store: Path, model_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    dest = store / model_name
    npy_path = dest / "embeddings.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(f"missing embeddings: {npy_path}")
    embeddings = np.load(npy_path)
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return embeddings, meta


def _stack_labels(values: pd.Series, task: str) -> np.ndarray:
    first = values.iloc[0]
    if isinstance(first, (list, tuple)):
        return np.asarray(values.tolist(), dtype=np.float32 if task == "regression" else np.int64)
    if isinstance(first, np.ndarray):
        return np.stack(values.to_numpy())
    if task == "regression":
        return values.to_numpy(dtype=np.float64)
    return values.to_numpy()


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    corr = np.corrcoef(y_true.astype(np.float64), y_pred.astype(np.float64))[0, 1]
    return float(corr)


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        result = spearmanr(y_true, y_pred)
        return float(result.statistic)
    except Exception:
        return float("nan")


def eval_split(task: str, model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if task == "regression":
        pred = model.predict(x)
        return {
            "mse": float(np.mean((pred - y) ** 2)),
            "r": _safe_pearson(y, pred),
            "p": _safe_spearman(y, pred),
        }
    if task == "classification":
        prob = model.predict_proba(x)[:, 1]
        return {
            "auroc": float(roc_auc_score(y, prob)),
            "auprc": float(average_precision_score(y, prob)),
        }
    if task == "multilabel":
        raw = model.predict_proba(x)
        prob = np.swapaxes(np.array(raw), 0, 1)[:, :, 1]
        return {
            "auroc": float(roc_auc_score(y, prob, average="micro")),
            "auprc": float(average_precision_score(y, prob, average="micro")),
        }
    raise ValueError(f"unsupported task {task}")


def fit_model(task: str, x: np.ndarray, y: np.ndarray, ridge_alphas: list[float]):
    if task == "regression":
        model = RidgeCV(alphas=ridge_alphas)
        try:
            model.fit(x, y)
        except ValueError:
            model = Ridge(solver="sag", alpha=ridge_alphas[0])
            model.fit(x, y)
        return model
    if task == "classification":
        model = LogisticRegression(max_iter=5000)
        model.fit(x, y)
        return model
    if task == "multilabel":
        model = MultiOutputClassifier(LogisticRegression(max_iter=5000), n_jobs=-1)
        model.fit(x, y)
        return model
    raise ValueError(f"unsupported task {task}")


def probe_one(
    *,
    sequences: pd.DataFrame,
    splits: pd.DataFrame,
    embeddings: np.ndarray,
    task: str,
    target_col: str,
    seeds: list[int],
    ridge_alphas: list[float],
) -> dict[str, Any]:
    if embeddings.shape[0] != len(sequences):
        raise ValueError(
            f"embedding rows {embeddings.shape[0]} != sequence rows {len(sequences)}"
        )
    indexed = sequences.set_index("row_index", drop=False)
    per_seed: dict[str, dict[str, float]] = {}
    subset = splits[splits["target_col"] == target_col]
    if subset.empty:
        raise ValueError(f"no frozen splits for target {target_col}")
    for seed in seeds:
        seed_splits = subset[subset["seed"] == int(seed)]
        parts: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        for split_name in ("train", "val", "test"):
            rows = seed_splits.loc[seed_splits["split"] == split_name, "row_index"].to_numpy()
            if rows.size == 0:
                continue
            frame = indexed.loc[rows]
            parts[split_name] = embeddings[frame["row_index"].to_numpy()]
            labels[split_name] = _stack_labels(frame[target_col], task)
        np.random.seed(int(seed))
        model = fit_model(task, parts["train"], labels["train"], ridge_alphas)
        metrics: dict[str, float] = {}
        for split_name in ("val", "test", "train"):
            if split_name not in parts:
                continue
            split_metrics = eval_split(task, model, parts[split_name], labels[split_name])
            for key, value in split_metrics.items():
                metrics[f"{split_name}_{key}"] = value
        per_seed[str(seed)] = metrics

    keys = sorted({key for metrics in per_seed.values() for key in metrics})
    summary: dict[str, Any] = {}
    n_seeds = len(per_seed)
    for key in keys:
        values = np.asarray([per_seed[seed][key] for seed in per_seed if key in per_seed[seed]], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[key] = {"mean": math.nan, "std": math.nan, "ci95": math.nan, "n": 0}
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        se = std / math.sqrt(values.size)
        summary[key] = {
            "mean": mean,
            "std": std,
            "ci95": 1.96 * se,
            "n": int(values.size),
        }
    return {"per_seed": per_seed, "summary": summary, "n_seeds": n_seeds, "task": task, "target_col": target_col}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--group", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    ds_registry = load_dataset_registry(cfg, project_root)
    dataset_ids = resolve_dataset_ids(ds_registry, group=args.group, datasets=args.datasets)
    model_names = args.models or list(cfg["models"]["names"])
    probe_cfg = cfg.get("probe", {})
    seeds = [int(seed) for seed in probe_cfg.get("seeds", [2541])]
    ridge_alphas = [float(x) for x in probe_cfg.get("ridge_alphas", [1e-3, 1e-2, 1e-1, 1.0, 10.0])]
    out_root = results_dir(cfg, project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset_id in dataset_ids:
        tables = table_dir(cfg, project_root, dataset_id)
        embed_root = embeddings_dir(cfg, project_root, dataset_id)
        if not (tables / "metadata.json").is_file():
            print(f"skip probe {dataset_id}: missing tables under {tables}", flush=True)
            continue
        sequences, splits, metadata = load_tables(tables)
        task = str(metadata.get("task") or ds_registry["datasets"][dataset_id].get("task", "regression"))
        target_cols = list(metadata.get("target_cols") or ["target"])
        for model_name in model_names:
            try:
                embeddings, embed_meta = _load_embeddings(embed_root, model_name)
            except FileNotFoundError:
                print(f"skip probe {model_name}/{dataset_id}: missing embeddings", flush=True)
                continue
            if embeddings.shape[0] != len(sequences):
                print(
                    f"skip probe {model_name}/{dataset_id}: "
                    f"N_embed={embeddings.shape[0]} N_seq={len(sequences)}",
                    flush=True,
                )
                continue
            model_out = out_root / dataset_id / model_name
            metrics_path = model_out / "metrics.json"
            if metrics_path.is_file():
                print(f"skip probe {model_name}/{dataset_id}: existing metrics", flush=True)
                continue
            model_out.mkdir(parents=True, exist_ok=True)
            combined: dict[str, Any] = {
                "dataset_id": dataset_id,
                "model": model_name,
                "embed_meta": embed_meta,
                "targets": {},
            }
            for target_col in target_cols:
                print(f"probe {model_name}/{dataset_id}/{target_col}", flush=True)
                combined["targets"][target_col] = probe_one(
                    sequences=sequences,
                    splits=splits,
                    embeddings=embeddings,
                    task=task,
                    target_col=target_col,
                    seeds=seeds,
                    ridge_alphas=ridge_alphas,
                )
            metrics_path.write_text(
                json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"wrote {model_out / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
