# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collect the per-sequence test tables into one table per model and task.

A model's test table is not always over the same sequences: the published
structRFM checkpoint is scored on the full test directory, the RIBOSPAN heads
on the sequences within the task's length limit. The ``shared`` columns hold
the means over the sequences every model of a task has scored, so that models
are compared on the same set.

INF is the geometric mean of a sequence's precision and recall, the score
structRFM reports beside F1. The figure shows the per-sequence INF on the
shared sequences for the models under ``figure.models`` of the experiment
configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.config import PROJECT_ROOT, load_yaml, registry_rows
from scripts.tasks import TASKS

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ssp_structrfm"
PUBLISHED_MODEL = "structRFM"
METRICS = {"f1s": "F1", "INF": "INF", "precision": "precision", "recall": "recall"}
# Task and title of each violin panel; the family boxes split FAMILY_TASK.
FIGURE_TASKS = (("RNAStrAlign", "ArchiveII"), ("bpRNA1m", "bpRNA TS0"))
FAMILY_TASK = "RNAStrAlign"
FIGURE_STEM = "secondary_structure_inf"


def _table(model: str, task: str) -> pd.DataFrame | None:
    directory = (
        OUTPUT_ROOT / "official" / model / task
        if model == PUBLISHED_MODEL
        else OUTPUT_ROOT / "heads" / model / task
    )
    found = sorted(directory.glob("test_metric_*.csv"))
    if not found:
        return None
    frame = pd.read_csv(found[0])
    frame["name"] = frame["name"].astype(str)
    frame["INF"] = np.sqrt(frame["precision"] * frame["recall"])
    return frame


def _shared_tables(task: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Every model's test table of ``task``, whole and on the shared sequences."""

    tables = {
        model: table
        for model in (row["name"] for row in registry_rows())
        if (table := _table(model, task)) is not None
    }
    if not tables:
        return {}, {}
    shared = set.intersection(*(set(table["name"]) for table in tables.values()))
    return tables, {
        model: table[table["name"].isin(shared)] for model, table in tables.items()
    }


def _figure() -> list[Path]:
    """Draw the per-sequence INF of the configured models that have every table."""

    from scripts.plotting import FAMILY_LABELS, plot_score_distributions

    experiment = load_yaml(PROJECT_ROOT / "configs" / "experiment.yaml")
    requested = (experiment.get("figure") or {}).get("models") or [
        row["name"] for row in registry_rows()
    ]
    # Sorted by name, so one family mask serves every model.
    shared = {
        task: {
            model: table.sort_values("name")
            for model, table in _shared_tables(task)[1].items()
        }
        for task, _ in FIGURE_TASKS
    }
    models = []
    for model in requested:
        if all(model in tables for tables in shared.values()):
            models.append(model)
        else:
            print(f"figure skips {model}: not every task has a test table", flush=True)
    if not models:
        return []
    family_tables = shared[FAMILY_TASK]
    family_of = family_tables[models[0]]["name"].str.split("_").str[0]
    families = [family for family in FAMILY_LABELS if (family_of == family).any()]
    families += sorted(set(family_of) - set(families))
    return plot_score_distributions(
        [
            (title, {model: shared[task][model]["INF"].to_numpy() for model in models})
            for task, title in FIGURE_TASKS
        ],
        "ArchiveII by RNA Family",
        {
            family: {
                model: family_tables[model]["INF"].to_numpy()[
                    (family_of == family).to_numpy()
                ]
                for model in models
            }
            for family in families
        },
        ylabel="INF ↑",
        number="mean",
        output_dir=OUTPUT_ROOT,
        stem=FIGURE_STEM,
    )


def summarize() -> list[Path]:
    """Write the summary tables and figure under the output root; return the paths."""

    rows: list[dict] = []
    for task, spec in TASKS.items():
        tables, shared = _shared_tables(task)
        for model, table in tables.items():
            on_shared = shared[model]
            row = {
                "model": model,
                "task": task,
                "test_set": spec["test"],
                "num_test": len(table),
            }
            row.update({label: table[column].mean() for column, label in METRICS.items()})
            row["shared_num_test"] = len(on_shared)
            row.update(
                {
                    f"shared_{label}": on_shared[column].mean()
                    for column, label in METRICS.items()
                }
            )
            rows.append(row)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.tsv"
    json_path = OUTPUT_ROOT / "summary.json"
    pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return [summary_path, json_path, *_figure()]
