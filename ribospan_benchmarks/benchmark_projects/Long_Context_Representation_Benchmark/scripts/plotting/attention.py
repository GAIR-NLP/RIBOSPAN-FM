# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Publication plots for aggregate RNABert attention diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .export import CLEANUP_EXTENSIONS, save_figure
from .model_style import (
    display_model_colors,
    display_model_name,
    model_colors,
    ordered_models,
)


BIN_COLUMNS = [
    "bin_0_8",
    "bin_9_32",
    "bin_33_128",
    "bin_129_512",
    "bin_513_1024",
    "bin_1025_2048",
    "bin_2049_4096",
    "bin_4097_plus",
]
BIN_LABELS = ["0-8", "9-32", "33-128", "129-512", "513-1024", "1025-2048", "2049-4096", "4097+"]
BIN_COLORS = [
    "#2F4858",
    "#33658A",
    "#86BBD8",
    "#758E4F",
    "#F6AE2D",
    "#F26419",
    "#B23A48",
    "#6D2E46",
]


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[str]:
    return save_figure(figure, output_dir, stem)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _ordered_models(frame: pd.DataFrame) -> list[str]:
    return ordered_models(frame["model_name"].dropna().astype(str))


def _model_colors(frame: pd.DataFrame) -> dict[str, Any]:
    return model_colors(frame["model_name"].dropna().astype(str))


def _display_models(frame: pd.DataFrame) -> list[str]:
    return [display_model_name(model) for model in _ordered_models(frame)]


def _with_display_model(
    frame: pd.DataFrame, column: str = "model_name"
) -> pd.DataFrame:
    output = frame.copy()
    output[column] = output[column].map(display_model_name)
    return output


def _prepare(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_name",
        "job_id",
        "length",
        "control_type",
        "layer",
        "head",
        "mean_distance",
        "normalized_mean_distance",
        *BIN_COLUMNS,
    }
    missing = required - set(summary)
    if missing:
        raise ValueError(
            f"attention_summary.csv misses plotting columns: {sorted(missing)}"
        )
    output = _numeric(
        summary,
        (
            "length",
            "layer",
            "head",
            "mean_distance",
            "normalized_mean_distance",
            "real_key_mass",
            *BIN_COLUMNS,
        ),
    )
    output = output.dropna(
        subset=["length", "layer", "head", "mean_distance"]
    )
    structured = output[
        output["control_type"].astype(str).str.lower() == "structured"
    ]
    return structured.copy() if not structured.empty else output


def _write_summary(frame: pd.DataFrame, output_dir: Path) -> str:
    work = frame.copy()
    work["mass_gt_512"] = work[
        ["bin_513_1024", "bin_1025_2048", "bin_2049_4096", "bin_4097_plus"]
    ].sum(axis=1)
    work["mass_gt_1024"] = work[
        ["bin_1025_2048", "bin_2049_4096", "bin_4097_plus"]
    ].sum(axis=1)
    work["mass_gt_2048"] = work[["bin_2049_4096", "bin_4097_plus"]].sum(
        axis=1
    )
    summary = (
        work.groupby(["length", "model_name"], as_index=False, dropna=False)
        .agg(
            job_count=("job_id", "nunique"),
            mean_distance=("mean_distance", "mean"),
            normalized_mean_distance=("normalized_mean_distance", "mean"),
            real_key_mass=("real_key_mass", "mean"),
            mass_gt_512=("mass_gt_512", "mean"),
            mass_gt_1024=("mass_gt_1024", "mean"),
            mass_gt_2048=("mass_gt_2048", "mean"),
        )
        .sort_values(["length", "model_name"])
    )
    path = output_dir / "00_summary_by_length_model.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return str(path)


def _plot_layer_mean(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    lengths = sorted(frame["length"].dropna().unique())
    models = _ordered_models(frame)
    raw_colors = _model_colors(frame)
    per_job = (
        frame.groupby(
            ["length", "model_name", "job_id", "layer"],
            as_index=False,
            dropna=False,
        )[["mean_distance", "normalized_mean_distance"]]
        .mean()
        .dropna()
    )
    figure, axes = plt.subplots(
        len(lengths),
        2,
        figsize=(16, max(5, 3.7 * len(lengths))),
        constrained_layout=True,
        squeeze=False,
    )
    for row, length in enumerate(lengths):
        length_data = per_job[per_job["length"] == length]
        for column, (metric, label) in enumerate(
            (
                ("mean_distance", "Raw mean distance"),
                ("normalized_mean_distance", "Normalized mean distance"),
            )
        ):
            axis = axes[row, column]
            for model in models:
                color = raw_colors[model]
                model_data = length_data[length_data["model_name"] == model]
                if model_data.empty:
                    continue
                sns.lineplot(
                    data=model_data,
                    x="layer",
                    y=metric,
                    estimator="mean",
                    errorbar=("ci", 95),
                    n_boot=2000,
                    seed=20260709,
                    label=display_model_name(model),
                    color=color,
                    linewidth=2.0,
                    ax=axis,
                )
            axis.set_title(f"Length {int(length)} | {label}")
            axis.set_xlabel("Layer")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", fontsize=7)
    paths = _save_figure(figure, output_dir, "01_layer_mean_distance_overview")
    plt.close(figure)
    return paths


def _plot_distance_bins(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt  # type: ignore

    models = _ordered_models(frame)
    lengths = sorted(frame["length"].dropna().unique())
    figure, axes = plt.subplots(
        len(lengths),
        len(models),
        figsize=(8.5 * len(models), max(5.8, 4.8 * len(lengths))),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, length in enumerate(lengths):
        length_data = frame[frame["length"] == length]
        for column_index, model in enumerate(models):
            axis = axes[row, column_index]
            model_data = length_data[length_data["model_name"] == model]
            if model_data.empty:
                axis.axis("off")
                continue
            per_layer = (
                model_data.groupby("layer", as_index=False)[BIN_COLUMNS]
                .mean()
                .sort_values("layer")
            )
            bottom = np.zeros(len(per_layer), dtype=float)
            for column, label, color in zip(
                BIN_COLUMNS, BIN_LABELS, BIN_COLORS
            ):
                values = per_layer[column].to_numpy(dtype=float)
                axis.bar(
                    per_layer["layer"],
                    values,
                    bottom=bottom,
                    label=label,
                    color=color,
                    width=0.85,
                )
                bottom += values
            axis.set_title(
                f"{int(length)} nt | {display_model_name(model)}"
            )
            axis.set_xlabel("Layer")
            axis.set_ylabel("Attention mass")
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=8,
        fontsize=8,
        frameon=True,
        framealpha=0.8,
    )
    figure.suptitle("Distance-bin attention mass by input length", fontsize=15)
    paths = _save_figure(
        figure,
        output_dir,
        "02_distance_bins_overview",
    )
    plt.close(figure)
    return paths


def _late_layer_convergence(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Attention_Heatmap-style middle-to-late layer changes per job."""

    work = frame.copy()
    work["mass_gt_1024"] = work[
        ["bin_1025_2048", "bin_2049_4096", "bin_4097_plus"]
    ].sum(axis=1)
    work["mass_gt_2048"] = work[
        ["bin_2049_4096", "bin_4097_plus"]
    ].sum(axis=1)
    metrics = [
        "mean_distance",
        "normalized_mean_distance",
        "mass_gt_1024",
        "mass_gt_2048",
    ]
    per_job = work.groupby(
        ["length", "model_name", "job_id", "layer"],
        as_index=False,
        dropna=False,
    )[metrics].mean()
    rows: list[dict[str, Any]] = []
    for (length, model, job_id), group in per_job.groupby(
        ["length", "model_name", "job_id"], dropna=False
    ):
        maximum_hidden_layer = int(group["layer"].max()) + 1
        if maximum_hidden_layer < 4:
            continue
        middle_start = max(1, int(math.ceil(maximum_hidden_layer * 0.25)))
        middle_end = max(
            middle_start, int(math.floor(maximum_hidden_layer * 0.50))
        )
        late_start = max(
            1, int(math.ceil(maximum_hidden_layer * 0.75))
        )
        hidden_layer = group["layer"] + 1
        middle = group[
            hidden_layer.between(middle_start, middle_end)
        ]
        late = group[
            hidden_layer.between(late_start, maximum_hidden_layer)
        ]
        if middle.empty or late.empty:
            continue
        row: dict[str, Any] = {
            "length": length,
            "model_name": model,
            "job_id": job_id,
            "middle_attention_layer_start": middle_start - 1,
            "middle_attention_layer_end": middle_end - 1,
            "late_attention_layer_start": late_start - 1,
            "late_attention_layer_end": maximum_hidden_layer - 1,
        }
        for metric in metrics:
            middle_mean = float(middle[metric].mean())
            late_mean = float(late[metric].mean())
            row[f"{metric}_middle_mean"] = middle_mean
            row[f"{metric}_late_mean"] = late_mean
            row[f"{metric}_late_minus_middle"] = late_mean - middle_mean
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_late_layer_convergence(
    frame: pd.DataFrame, output_dir: Path
) -> list[str]:
    convergence = _late_layer_convergence(frame)
    path = output_dir / "00_late_layer_attention_convergence.csv"
    convergence.to_csv(path, index=False)
    outputs = [str(path)]
    if convergence.empty:
        return outputs
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    palette = display_model_colors(frame["model_name"].dropna().astype(str))
    order = _display_models(frame)
    metric_columns = {
        "normalized_mean_distance_late_minus_middle": "Normalized\nmean distance",
        "mass_gt_1024_late_minus_middle": "Attention mass\n>1024 nt",
        "mass_gt_2048_late_minus_middle": "Attention mass\n>2048 nt",
    }
    long = _with_display_model(convergence).melt(
        id_vars=["length", "model_name", "job_id"],
        value_vars=list(metric_columns),
        var_name="metric",
        value_name="late_minus_middle",
    )
    long["metric"] = long["metric"].map(metric_columns)
    lengths = sorted(long["length"].dropna().unique())
    columns = min(2, len(lengths))
    rows = math.ceil(len(lengths) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10 * columns, 5.5 * rows),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
    )
    for axis, length in zip(axes.flat, lengths):
        selected = long[long["length"] == length]
        sns.barplot(
            data=selected,
            x="metric",
            y="late_minus_middle",
            hue="model_name",
            hue_order=order,
            palette=palette,
            estimator="mean",
            errorbar=("ci", 95),
            capsize=0.12,
            err_kws={"linewidth": 1.1, "color": "#444444"},
            edgecolor="white",
            ax=axis,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(
            f"Input length {int(length)} nt | late blocks 23–31 minus middle 7–15"
        )
        axis.set_xlabel("")
        axis.set_ylabel("Late quarter − middle quarter")
        axis.grid(True, alpha=0.25)
        axis.legend(
            title="Model",
            frameon=True,
            framealpha=0.8,
            facecolor="white",
            edgecolor="#BBBBBB",
        )
        axis.margins(y=0.08)
    for axis in list(axes.flat)[len(lengths) :]:
        axis.axis("off")
    figure.suptitle(
        "Late-layer attention convergence (negative = selective re-localization)",
        fontsize=14,
    )
    outputs.extend(
        _save_figure(
            figure, output_dir, "03_late_layer_attention_convergence"
        )
    )
    plt.close(figure)
    return outputs


def plot_attention_diagnostics(
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Write Attention_Heatmap-style aggregate distance diagnostics."""

    if summary.empty:
        return []
    try:
        import seaborn as sns  # type: ignore
    except ImportError:
        return []

    sns.set_theme(style="whitegrid")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in CLEANUP_EXTENSIONS:
        for path in output_dir.glob(
            f"02_distance_bins_length_*.{extension}"
        ):
            path.unlink()
    frame = _prepare(summary)
    paths = [_write_summary(frame, output_dir)]
    paths.extend(_plot_layer_mean(frame, output_dir))
    paths.extend(_plot_distance_bins(frame, output_dir))
    paths.extend(_plot_late_layer_convergence(frame, output_dir))
    return paths
