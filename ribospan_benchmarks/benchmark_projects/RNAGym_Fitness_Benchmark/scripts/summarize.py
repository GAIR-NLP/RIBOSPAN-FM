# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collect per-assay metrics and RNAGym-style type-mean overall scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (  # noqa: E402
    load_dataset_registry,
    load_experiment_config,
    output_root,
    results_dir,
)


RNA_TYPES = ("mRNA-splicing", "mRNA-coding", "tRNA", "Aptamer", "Ribozyme")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mean(values: pd.Series) -> float:
    part = pd.to_numeric(values, errors="coerce").dropna()
    if part.empty:
        return float("nan")
    return float(part.mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    registry = load_dataset_registry(cfg, project_root)
    metrics_root = results_dir(cfg, project_root)
    summary_root = output_root(cfg, project_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(metrics_root.glob("*/*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        dataset_id = payload.get("dataset_id") or metrics_path.parent.parent.name
        spec = registry.get("datasets", {}).get(dataset_id, {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "rna_type": payload.get("rna_type") or spec.get("rna_type"),
                "model": payload.get("model") or metrics_path.parent.name,
                "Spearman": metrics.get("Spearman"),
                "AUC": metrics.get("AUC"),
                "MCC": metrics.get("MCC"),
                "n_scored": metrics.get("n_scored"),
                "n": metrics.get("n"),
                "single_Spearman": metrics.get("single_Spearman"),
                "multiple_Spearman": metrics.get("multiple_Spearman"),
            }
        )
    frame = pd.DataFrame(rows)
    summary_root.mkdir(parents=True, exist_ok=True)
    assay_path = summary_root / "summary_by_assay.tsv"
    type_path = summary_root / "summary_by_rna_type.tsv"
    overall_path = summary_root / "summary_overall.tsv"
    json_path = summary_root / "summary.json"
    if frame.empty:
        print(f"no metrics under {metrics_root}", flush=True)
        _write_json(json_path, [])
        assay_path.write_text("dataset_id\trna_type\tmodel\tSpearman\n", encoding="utf-8")
        return 0
    frame.sort_values(["rna_type", "dataset_id", "model"]).to_csv(assay_path, sep="\t", index=False)

    type_rows = []
    for (rna_type, model), part in frame.groupby(["rna_type", "model"]):
        type_rows.append(
            {
                "rna_type": rna_type,
                "model": model,
                "n_assays": int(part["Spearman"].notna().sum()),
                "Spearman": _mean(part["Spearman"]),
                "AUC": _mean(part["AUC"]),
                "MCC": _mean(part["MCC"]),
            }
        )
    type_frame = pd.DataFrame(type_rows)
    type_frame.to_csv(type_path, sep="\t", index=False)

    overall_rows = []
    for model, part in type_frame.groupby("model"):
        present = part.set_index("rna_type")
        type_s, type_a, type_m = [], [], []
        for rna_type in RNA_TYPES:
            if rna_type not in present.index:
                continue
            row = present.loc[rna_type]
            if pd.notna(row["Spearman"]):
                type_s.append(float(row["Spearman"]))
            if pd.notna(row["AUC"]):
                type_a.append(float(row["AUC"]))
            if pd.notna(row["MCC"]):
                type_m.append(float(row["MCC"]))
        overall_rows.append(
            {
                "model": model,
                "n_types": len(type_s),
                "n_assays": int(part["n_assays"].sum()),
                "Spearman": float(sum(type_s) / len(type_s)) if type_s else float("nan"),
                "AUC": float(sum(type_a) / len(type_a)) if type_a else float("nan"),
                "MCC": float(sum(type_m) / len(type_m)) if type_m else float("nan"),
            }
        )
    overall_frame = pd.DataFrame(overall_rows)
    model_order = list(cfg.get("models", {}).get("names") or [])
    if model_order and not overall_frame.empty:
        overall_frame["model"] = pd.Categorical(overall_frame["model"], categories=model_order, ordered=True)
        overall_frame = overall_frame.sort_values("model")
    overall_frame.to_csv(overall_path, sep="\t", index=False)
    _write_json(
        json_path,
        {
            "by_assay": frame.to_dict(orient="records"),
            "by_rna_type": type_frame.to_dict(orient="records"),
            "overall": overall_frame.to_dict(orient="records"),
        },
    )
    print(f"wrote {assay_path}", flush=True)
    print(f"wrote {type_path}", flush=True)
    print(f"wrote {overall_path}", flush=True)
    if not overall_frame.empty:
        print("\noverall (mean of RNA-type means)", flush=True)
        print(overall_frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
