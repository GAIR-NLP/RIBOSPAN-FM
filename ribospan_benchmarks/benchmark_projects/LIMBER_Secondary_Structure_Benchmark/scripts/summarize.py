# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collect the per-run ``result.json`` files into the tables LIMBER reports.

bpRNA TS0 and Rivas TestSetB are reported as the mean and median F1 over test
sequences. ArchiveII leave-one-family-out is reported as LIMBER does: the mean
F1 pooled over the test sequences of all held-out families.

The figure shows the per-sequence F1 behind those numbers for the models under
``figure.models`` of the experiment configuration.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.config import model_rows, output_dir

SINGLE_SPLIT_TASKS = ("bprna_ts0", "rivas_testsetb")
ARCHIVE_TASK = "archiveII_lfo"
FIGURE_TITLES = {"bprna_ts0": "bpRNA TS0", "rivas_testsetb": "Rivas TestSetB"}
FIGURE_STEM = "secondary_structure_flexible_f1"
SUMMARY_FIELDS = [
    "model",
    *(
        f"{task}_{field}"
        for task in SINGLE_SPLIT_TASKS
        for field in ("num_test", "mean_F1", "median_F1")
    ),
    f"{ARCHIVE_TASK}_families",
    f"{ARCHIVE_TASK}_num_test",
    f"{ARCHIVE_TASK}_pooled_mean_F1",
]
FAMILY_FIELDS = [
    "model",
    "family",
    "num_test",
    "mean_F1",
    "median_F1",
    "mean_precision",
    "mean_recall",
]


def _output_root(config: dict[str, Any], root: Path) -> Path:
    relative = Path(config.get("output_root", "outputs/limber_ss"))
    return relative if relative.is_absolute() else root / relative


def _result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _figure(
    config: dict[str, Any], root: Path, families: tuple[str, ...]
) -> list[Path]:
    """Draw the per-sequence F1 of the configured models that have every result."""

    from scripts.plotting import FAMILY_LABELS, plot_score_distributions

    requested = (config.get("figure") or {}).get("models") or [
        row["name"] for row in model_rows(config, root)
    ]
    scores: dict[tuple[str, str | None], dict[str, list[float]]] = {}
    for model in requested:
        runs = {
            (task, family): _result(
                output_dir(config, root, model, task, family) / "result.json"
            )
            for task, family in [
                *((task, None) for task in SINGLE_SPLIT_TASKS),
                *((ARCHIVE_TASK, family) for family in families),
            ]
        }
        if any(result is None for result in runs.values()):
            print(f"figure skips {model}: not every task has a result", flush=True)
            continue
        for key, result in runs.items():
            scores.setdefault(key, {})[model] = result["per_seq_f1"]
    if not scores:
        return []
    ordered = [family for family in FAMILY_LABELS if family in families]
    ordered += [family for family in families if family not in FAMILY_LABELS]
    return plot_score_distributions(
        [(FIGURE_TITLES[task], scores[(task, None)]) for task in SINGLE_SPLIT_TASKS],
        "ArchiveII Leave-one-family-out",
        {family: scores[(ARCHIVE_TASK, family)] for family in ordered},
        ylabel="Flexible F1 ↑",
        number="median",
        output_dir=_output_root(config, root),
        stem=FIGURE_STEM,
    )


def summarize(
    config: dict[str, Any], root: Path, families: tuple[str, ...]
) -> list[Path]:
    """Write the summary tables and figure under the output root; return the paths."""

    summary_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for model in (row["name"] for row in model_rows(config, root)):
        summary: dict[str, Any] = {"model": model}
        for task in SINGLE_SPLIT_TASKS:
            result = _result(output_dir(config, root, model, task) / "result.json")
            if result is None:
                continue
            for field in ("num_test", "mean_F1", "median_F1"):
                summary[f"{task}_{field}"] = result[field]
        held_out = [
            result
            for family in families
            if (
                result := _result(
                    output_dir(config, root, model, ARCHIVE_TASK, family)
                    / "result.json"
                )
            )
            is not None
        ]
        for result in held_out:
            family_rows.append({"model": model, **result})
        if held_out:
            sequences = sum(int(result["num_test"]) for result in held_out)
            summary[f"{ARCHIVE_TASK}_families"] = len(held_out)
            summary[f"{ARCHIVE_TASK}_num_test"] = sequences
            summary[f"{ARCHIVE_TASK}_pooled_mean_F1"] = (
                sum(result["mean_F1"] * int(result["num_test"]) for result in held_out)
                / sequences
            )
        if len(summary) > 1:
            summary_rows.append(summary)

    destination = _output_root(config, root)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.tsv"
    family_path = destination / "summary_archiveII_by_family.tsv"
    json_path = destination / "summary.json"
    _write_tsv(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_tsv(family_path, FAMILY_FIELDS, family_rows)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "archiveII_by_family": [
                    {field: row.get(field) for field in FAMILY_FIELDS}
                    for row in family_rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [summary_path, family_path, json_path, *_figure(config, root, families)]
