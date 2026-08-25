# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Publication plots for cosine contextualization and causal diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .export import CLEANUP_EXTENSIONS, save_figure
from .model_style import (
    display_model_colors,
    display_model_name,
    ordered_models,
)


CONTEXT_COMPONENTS = {
    "same_context_baseline_cos": "Same-context baseline",
    "cross_context_same_base_cos": "Cross-context same-base",
    "within_motif_same_base_cos": "Within structured window",
    "within_background_same_base_cos": "Within background",
}
COMPONENT_COLORS = {
    "Same-context baseline": "#0072B2",
    "Cross-context same-base": "#D55E00",
    "Within structured window": "#009E73",
    "Within background": "#CC79A7",
}
PRIMARY_RELATIVE_DISTAL_THRESHOLD = 0.75
LENGTH_BARPLOT_KWARGS = {
    "estimator": "mean",
    "errorbar": ("ci", 95),
    "n_boot": 2000,
    "seed": 20260709,
    "capsize": 0.12,
    "err_kws": {"linewidth": 1.1, "color": "#444444"},
    "edgecolor": "white",
}
TABLE_METRIC_COLUMNS = (
    "delta_context_separation_structured_vs_native",
    "cross_context_same_base_cos_structured",
    "relative_distal_075",
)


def _ordered_models(values: Iterable[object]) -> list[str]:
    return ordered_models(values)


def _display_models(values: Iterable[object]) -> list[str]:
    return [display_model_name(model) for model in ordered_models(values)]


def _with_display_model(
    frame: pd.DataFrame, column: str = "model"
) -> pd.DataFrame:
    output = frame.copy()
    output[column] = output[column].map(display_model_name)
    return output


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[str]:
    return save_figure(figure, output_dir, stem)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _prepare_context(
    context: pd.DataFrame, *, structured_only: bool = True
) -> pd.DataFrame:
    required = {
        "model",
        "length",
        "layer",
        "base",
        "control_type",
        "metric_variant",
        "context_separation",
        *CONTEXT_COMPONENTS,
    }
    missing = required - set(context)
    if missing:
        raise ValueError(
            f"cosine_context.csv misses plotting columns: {sorted(missing)}"
        )
    output = context[
        context["metric_variant"].astype(str).str.lower() == "raw"
    ].copy()
    output["control_type"] = output["control_type"].astype(str).str.lower()
    if structured_only:
        output = output[output["control_type"] == "structured"].copy()
    output = _numeric(
        output,
        ("length", "layer", "context_separation", *CONTEXT_COMPONENTS),
    )
    return output.dropna(subset=["model", "length", "layer", "base"])


def _prepare_diffusion(diffusion: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "length",
        "layer",
        "control_type",
        "metric_variant",
        "bin_index",
        "distance_bin",
        "mean_distance",
        "mean",
        "local_peak",
        "far_field",
        "relative_distal",
        "relative_distal_threshold",
        "relative_distal_025",
        "relative_distal_050",
        "relative_distal_075",
        "leakage_auc",
    }
    missing = required - set(diffusion)
    if missing:
        raise ValueError(
            f"cosine_diffusion.csv misses plotting columns: {sorted(missing)}"
        )
    output = diffusion[
        (
            diffusion["control_type"]
            .astype(str)
            .str.lower()
            .isin(["native", "none"])
        )
        & (diffusion["metric_variant"].astype(str).str.lower() == "raw")
    ].copy()
    output = _numeric(
        output,
        (
            "length",
            "layer",
            "bin_index",
            "mean_distance",
            "mean",
            "local_peak",
            "far_field",
            "relative_distal",
            "relative_distal_threshold",
            "relative_distal_025",
            "relative_distal_050",
            "relative_distal_075",
            "leakage_auc",
            "anisotropy_pair_baseline",
        ),
    )
    return output.dropna(subset=["model", "length", "layer", "bin_index"])


def _write_summaries(
    context: pd.DataFrame,
    diffusion: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_columns = ["context_separation", *CONTEXT_COMPONENTS]
    context_summary = (
        context.groupby(
            ["model", "length", "layer", "base"],
            as_index=False,
            dropna=False,
        )[context_columns]
        .mean()
        .sort_values(["model", "length", "layer", "base"])
    )
    context_path = output_dir / "00_context_layer_base_summary.csv"
    context_summary.to_csv(context_path, index=False)

    endpoints = [
        column
        for column in (
            "local_peak",
            "far_field",
            "relative_distal",
            "relative_distal_025",
            "relative_distal_050",
            "relative_distal_075",
            "leakage_auc",
            "anisotropy_pair_baseline",
        )
        if column in diffusion
    ]
    diffusion_layer = (
        diffusion.groupby(
            ["model", "length", "layer"],
            as_index=False,
            dropna=False,
        )[endpoints]
        .mean()
        .sort_values(["model", "length", "layer"])
    )
    layer_path = output_dir / "00_diffusion_layer_summary.csv"
    diffusion_layer.to_csv(layer_path, index=False)

    distance_summary = (
        diffusion.groupby(
            [
                "model",
                "length",
                "layer",
                "bin_index",
                "distance_bin",
                "mean_distance",
            ],
            as_index=False,
            dropna=False,
        )["mean"]
        .mean()
        .sort_values(["model", "length", "layer", "bin_index"])
    )
    distance_path = output_dir / "00_diffusion_distance_summary.csv"
    distance_summary.to_csv(distance_path, index=False)
    return [str(context_path), str(layer_path), str(distance_path)]


def _write_control_summary(
    context: pd.DataFrame, output_dir: Path
) -> str:
    columns = ["context_separation", *CONTEXT_COMPONENTS]
    summary = (
        context.groupby(
            ["model", "length", "control_type", "layer", "base"],
            as_index=False,
            dropna=False,
        )[columns]
        .mean()
        .sort_values(
            ["model", "length", "control_type", "layer", "base"]
        )
    )
    path = output_dir / "00_context_layer_base_by_control.csv"
    summary.to_csv(path, index=False)
    return str(path)


def _plot_joint_trajectory(
    context: pd.DataFrame,
    diffusion: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
    title_suffix: str,
) -> list[str]:
    """Plot layer trajectories using one common outer-distance region."""

    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.collections import LineCollection  # type: ignore
    from matplotlib.colors import Normalize  # type: ignore

    lengths = sorted(
        set(context["length"].dropna())
        & set(diffusion["length"].dropna())
    )
    models = _ordered_models(
        set(context["model"].dropna().astype(str))
        & set(diffusion["model"].dropna().astype(str))
    )
    trajectories: dict[tuple[float, str], pd.DataFrame] = {}
    for length in lengths:
        for model in models:
            context_selected = context[
                (context["length"] == length)
                & (context["model"] == model)
            ]
            diffusion_selected = diffusion[
                (diffusion["length"] == length)
                & (diffusion["model"] == model)
            ]
            if context_selected.empty or diffusion_selected.empty:
                continue
            context_layer = (
                context_selected.groupby("layer", as_index=False)[
                    "context_separation"
                ].mean()
            )
            diffusion_layer = (
                diffusion_selected.groupby("layer", as_index=False)[
                    "relative_distal_075"
                ].mean()
            )
            merged = (
                context_layer.merge(diffusion_layer, on="layer")
                .dropna()
                .sort_values("layer")
            )
            if len(merged) >= 2:
                trajectories[(float(length), model)] = merged
    if not trajectories:
        return []
    plot_lengths = sorted({key[0] for key in trajectories})
    minimum_layer = min(
        float(frame["layer"].min()) for frame in trajectories.values()
    )
    maximum_layer = max(
        float(frame["layer"].max()) for frame in trajectories.values()
    )
    norm = Normalize(vmin=minimum_layer, vmax=maximum_layer)
    columns = len(models)
    rows = len(plot_lengths)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.2 * columns, 6.2 * rows),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    scatter = None
    active_axes = []
    for row, length in enumerate(plot_lengths):
        for column, model in enumerate(models):
            axis = axes[row, column]
            merged = trajectories.get((float(length), model))
            if merged is None:
                axis.axis("off")
                continue
            active_axes.append(axis)
            x = merged["context_separation"].to_numpy(dtype=float)
            y = merged["relative_distal_075"].to_numpy(dtype=float)
            layers = merged["layer"].to_numpy(dtype=float)
            points = np.column_stack([x, y])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(
                segments,
                cmap="viridis",
                norm=norm,
                linewidth=2.0,
                alpha=0.8,
            )
            collection.set_array(layers[:-1])
            axis.add_collection(collection)
            scatter = axis.scatter(
                x,
                y,
                c=layers,
                cmap="viridis",
                norm=norm,
                s=48.0,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            annotation_layers = {
                int(layers[0]),
                int(layers[-1]),
                *[int(value) for value in layers if int(value) % 8 == 0],
                int(
                    merged.loc[
                        merged["context_separation"].idxmax(), "layer"
                    ]
                ),
                int(
                    merged.loc[
                        merged["relative_distal_075"].idxmax(), "layer"
                    ]
                ),
            }
            for _, point in merged.iterrows():
                if int(point["layer"]) in annotation_layers:
                    axis.annotate(
                        str(int(point["layer"])),
                        (
                            point["context_separation"],
                            point["relative_distal_075"],
                        ),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
            axis.autoscale()
            axis.set_title(f"{int(length)} nt | {display_model_name(model)}")
            if row == rows - 1:
                axis.set_xlabel(
                    "Context Separation (higher = more contextualized)"
                )
            else:
                axis.set_xlabel("")
            if column == 0:
                axis.set_ylabel("Relative distal diffusion")
            else:
                axis.set_ylabel("")
            axis.grid(True, alpha=0.25)
    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            ax=active_axes,
            fraction=0.018,
            pad=0.015,
            aspect=45,
        )
        colorbar.set_label("Hidden layer")
    figure.suptitle(
        f"Layer trajectory: contextualization versus fixed-relative distal diffusion\n"
        f"Common cutoff: normalized motif distance ≥ "
        f"{PRIMARY_RELATIVE_DISTAL_THRESHOLD:.2f} | {title_suffix}",
        fontsize=15,
    )
    paths = _save_figure(
        figure, output_dir, f"{stem_prefix}__02_context_diffusion_trajectory"
    )
    plt.close(figure)
    return paths


def _plot_context_geometry(
    comparison: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
    title_suffix: str,
) -> list[str]:
    """Plot every model in one shared context-geometry layout."""

    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    if comparison.empty:
        return []
    lengths = sorted(comparison["length"].dropna().unique())
    models = _ordered_models(comparison["model"].dropna().unique())
    figure, axes = plt.subplots(
        len(lengths),
        len(models),
        figsize=(8 * len(models), max(5.5, 4.2 * len(lengths))),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, length in enumerate(lengths):
        for column, model in enumerate(models):
            axis = axes[row, column]
            model_frame = comparison[
                (comparison["length"] == length)
                & (comparison["model"] == model)
            ]
            if model_frame.empty:
                axis.axis("off")
                continue
            component_columns = [
                f"{metric}_structured"
                for metric in CONTEXT_COMPONENTS
                if f"{metric}_structured" in model_frame
            ]
            component = model_frame.melt(
                id_vars=["pair_group_id", "layer", "base"],
                value_vars=component_columns,
                var_name="component",
                value_name="cosine",
            ).dropna()
            component["component"] = (
                component["component"]
                .str.replace(r"_structured$", "", regex=True)
                .map(CONTEXT_COMPONENTS)
            )
            component = (
                component.groupby(
                    ["pair_group_id", "layer", "component"],
                    as_index=False,
                )["cosine"]
                .mean()
            )
            sns.lineplot(
                data=component,
                x="layer",
                y="cosine",
                hue="component",
                palette=COMPONENT_COLORS,
                estimator="mean",
                errorbar=("ci", 95),
                linewidth=2.0,
                ax=axis,
            )
            axis.set_title(
                f"{int(length)} nt | {display_model_name(model)}"
            )
            axis.set_xlabel("Hidden layer (0 = embedding)")
            axis.set_ylabel("Cosine similarity")
            axis.grid(True, alpha=0.25)
            legend = axis.get_legend()
            if row != 0 or column != 0:
                if legend is not None:
                    legend.remove()
    figure.suptitle(title_suffix, fontsize=15)
    paths = _save_figure(
        figure, output_dir, f"{stem_prefix}__01_context_geometry"
    )
    plt.close(figure)
    return paths


def _plot_centroid_margins(
    trajectories: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
    title_suffix: str,
) -> list[str]:
    """Compare base-balanced leave-one-out centroid margins across models."""

    if trajectories.empty:
        return []
    required = {
        "model",
        "length",
        "control_type",
        "layer",
        "region",
        "base",
        "centroid_margin",
    }
    if required - set(trajectories):
        return []
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    frame = trajectories[
        trajectories["control_type"].astype(str).str.lower() == "structured"
    ].copy()
    if frame.empty:
        return []
    frame = _numeric(
        frame,
        ("layer", "centroid_margin"),
    )
    lengths = sorted(frame["length"].dropna().unique())
    models = _ordered_models(frame["model"].dropna().unique())
    figure, axes = plt.subplots(
        len(lengths),
        len(models),
        figsize=(8 * len(models), max(5.5, 4.2 * len(lengths))),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    margin_limits = (
        float(frame["centroid_margin"].min()),
        float(frame["centroid_margin"].max()),
    )
    for row, length in enumerate(lengths):
        for column, model in enumerate(models):
            axis = axes[row, column]
            model_frame = frame[
                (frame["length"] == length)
                & (frame["model"] == model)
            ]
            if model_frame.empty:
                axis.axis("off")
                continue
            base_balanced = (
                model_frame.groupby(
                    ["job_id", "layer", "region", "base"],
                    as_index=False,
                )["centroid_margin"]
                .mean()
                .dropna()
                .groupby(
                    ["job_id", "layer", "region"], as_index=False
                )["centroid_margin"]
                .mean()
            )
            sns.lineplot(
                data=base_balanced,
                x="layer",
                y="centroid_margin",
                hue="region",
                estimator="mean",
                errorbar=("ci", 95),
                palette={"pattern": "#F26419", "background": "#33658A"},
                ax=axis,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(
                f"{int(length)} nt | {display_model_name(model)}"
            )
            axis.set_xlabel("Hidden layer")
            axis.set_ylabel("Structured − background centroid margin")
            axis.set_ylim(*margin_limits)
            axis.grid(True, alpha=0.25)
            legend = axis.get_legend()
            if row == 0 and column == 0:
                axis.legend(
                    fontsize=8,
                    title_fontsize=8,
                    frameon=True,
                    framealpha=0.75,
                    facecolor="white",
                    edgecolor="#BBBBBB",
                )
            elif legend is not None:
                legend.remove()
    figure.suptitle(
        f"Base-balanced leave-one-out centroid margin\n{title_suffix}",
        fontsize=15,
    )
    paths = _save_figure(
        figure, output_dir, f"{stem_prefix}__03_centroid_margin"
    )
    plt.close(figure)
    return paths


def _plot_attention_similarity_coupling(
    association: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
) -> list[str]:
    """Show every model in one aligned attention–geometry layout."""

    required = {
        "model",
        "length",
        "hidden_layer",
        "normalized_mean_distance",
        "cross_context_same_base_cos_structured",
        "context_separation_structured",
    }
    if association.empty or required - set(association):
        return []
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.colors import Normalize  # type: ignore

    metrics = [
        "normalized_mean_distance",
        "cross_context_same_base_cos_structured",
        "context_separation_structured",
    ]
    lengths = sorted(association["length"].dropna().unique())
    models = _ordered_models(association["model"].dropna().unique())
    layer_frames: dict[tuple[float, str], pd.DataFrame] = {}
    for length in lengths:
        for model in models:
            layer = (
                association[
                    (association["length"] == length)
                    & (association["model"] == model)
                ]
                .groupby("hidden_layer", as_index=False)[metrics]
                .mean()
                .dropna()
                .sort_values("hidden_layer")
            )
            if len(layer) >= 2:
                layer_frames[(float(length), model)] = layer
    if not layer_frames:
        return []
    norm = Normalize(
        vmin=min(float(frame["hidden_layer"].min()) for frame in layer_frames.values()),
        vmax=max(float(frame["hidden_layer"].max()) for frame in layer_frames.values()),
    )
    metric_specs = (
        (
            "cross_context_same_base_cos_structured",
            "Cross-region similarity",
            "Cosine similarity",
        ),
        (
            "context_separation_structured",
            "Context discrimination",
            "Separation score",
        ),
    )
    normalized_values = np.concatenate(
        [
            frame["normalized_mean_distance"].to_numpy(dtype=float)
            for frame in layer_frames.values()
        ]
    )
    x_padding = max(
        0.01,
        0.04
        * (
            float(np.nanmax(normalized_values))
            - float(np.nanmin(normalized_values))
        ),
    )
    y_limits = {}
    for metric, _, _ in metric_specs:
        values = np.concatenate(
            [
                frame[metric].to_numpy(dtype=float)
                for frame in layer_frames.values()
            ]
        )
        padding = max(
            0.01,
            0.04 * (float(np.nanmax(values)) - float(np.nanmin(values))),
        )
        y_limits[metric] = (
            float(np.nanmin(values)) - padding,
            float(np.nanmax(values)) + padding,
        )
    figure = plt.figure(
        figsize=(5.3 * len(models) * len(metric_specs), 3.8 * len(lengths))
    )
    spacer_width = 0.10
    grid = figure.add_gridspec(
        len(lengths),
        len(models) * len(metric_specs) + 1,
        width_ratios=(
            [1.0] * len(models)
            + [spacer_width]
            + [1.0] * len(models)
        ),
        left=0.07,
        right=0.92,
        bottom=0.05,
        top=0.953,
        hspace=0.08,
        wspace=0.08,
    )
    axes = np.empty(
        (len(lengths), len(models) * len(metric_specs)),
        dtype=object,
    )
    for row in range(len(lengths)):
        for metric_index in range(len(metric_specs)):
            for model_index in range(len(models)):
                column = metric_index * len(models) + model_index
                grid_column = (
                    model_index
                    if metric_index == 0
                    else len(models) + 1 + model_index
                )
                axes[row, column] = figure.add_subplot(
                    grid[row, grid_column]
                )
    scatter = None
    for row, length in enumerate(lengths):
        for metric_index, (metric, _, ylabel) in enumerate(metric_specs):
            for model_index, model in enumerate(models):
                column = metric_index * len(models) + model_index
                axis = axes[row, column]
                layer = layer_frames.get((float(length), model))
                if layer is None:
                    axis.axis("off")
                    continue
                scatter = axis.scatter(
                layer["normalized_mean_distance"],
                layer[metric],
                c=layer["hidden_layer"],
                cmap="viridis",
                norm=norm,
                s=42,
                edgecolor="white",
                linewidth=0.4,
            )
                axis.plot(
                    layer["normalized_mean_distance"],
                    layer[metric],
                    color="#777777",
                    linewidth=0.8,
                    alpha=0.5,
                )
                axis.set_xlim(
                    float(np.nanmin(normalized_values)) - x_padding,
                    float(np.nanmax(normalized_values)) + x_padding,
                )
                axis.set_ylim(*y_limits[metric])
                if row == 0:
                    axis.set_title(display_model_name(model), fontsize=9)
                if row == len(lengths) - 1:
                    axis.set_xlabel("Attention normalized mean distance")
                else:
                    axis.tick_params(axis="x", labelbottom=False)
                if model_index == 0:
                    axis.set_ylabel(ylabel, labelpad=10)
                else:
                    axis.set_ylabel("")
                    axis.tick_params(axis="y", labelleft=False)
                if column == 0:
                    axis.text(
                        -0.22,
                        0.5,
                        f"{int(length)} nt",
                        transform=axis.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=9,
                        fontweight="bold",
                    )
                axis.grid(True, alpha=0.25)
    if scatter is not None:
        colorbar_axis = figure.add_axes([0.945, 0.15, 0.012, 0.68])
        colorbar = figure.colorbar(
            scatter,
            cax=colorbar_axis,
        )
        colorbar.set_label("Hidden layer")
    left_group_center = (
        axes[0, 0].get_position().x0
        + axes[0, len(models) - 1].get_position().x1
    ) / 2.0
    right_group_center = (
        axes[0, len(models)].get_position().x0
        + axes[0, len(models) * 2 - 1].get_position().x1
    ) / 2.0
    figure.text(
        left_group_center,
        0.974,
        "Cross-region similarity",
        ha="center",
        va="top",
        fontsize=11,
    )
    figure.text(
        right_group_center,
        0.974,
        "Context discrimination",
        ha="center",
        va="top",
        fontsize=11,
    )
    figure.suptitle(
        "Attention–representation coupling",
        y=0.997,
        fontsize=15,
    )
    paths = _save_figure(
        figure,
        output_dir,
        f"{stem_prefix}__04_attention_similarity_coupling",
    )
    plt.close(figure)
    return paths


def _with_length_label(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["length_label"] = output["length"].map(lambda value: str(int(value)))
    return output


def _draw_length_barplot(
    axis: Any,
    data: pd.DataFrame,
    metric: str,
    *,
    display_models: list[str],
    palette: dict[str, str],
    barplot_kwargs: dict[str, Any] | None = None,
) -> None:
    import seaborn as sns  # type: ignore

    kwargs = dict(LENGTH_BARPLOT_KWARGS)
    if barplot_kwargs:
        kwargs.update(barplot_kwargs)
    sns.barplot(
        data=data,
        x="length_label",
        y=metric,
        hue="model",
        hue_order=display_models,
        palette=palette,
        ax=axis,
        **kwargs,
    )


def _style_length_bar_axis(
    axis: Any,
    *,
    title: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    axis.set_title(title)
    axis.set_xlabel("Input length (nt)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.margins(y=0.08)
    legend = axis.get_legend()
    if show_legend:
        axis.legend(
            title="Model",
            frameon=True,
            framealpha=0.8,
            fontsize=8,
        )
    elif legend is not None:
        legend.remove()


def _final_context_endpoints(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return final-layer pair-level ΔCS and structured C_SB."""

    metrics = [
        column
        for column in (
            "delta_context_separation_structured_vs_native",
            "cross_context_same_base_cos_structured",
        )
        if column in comparison
    ]
    required = {"model", "length", "pair_group_id", "layer"}
    if comparison.empty or required - set(comparison) or not metrics:
        return pd.DataFrame()
    comparison = comparison.copy()
    final_layer = comparison.groupby(["model", "length"])["layer"].transform(
        "max"
    )
    final_context = comparison[comparison["layer"] == final_layer]
    return (
        final_context.groupby(
            ["model", "length", "pair_group_id"], as_index=False
        )[metrics]
        .mean()
        .sort_values(["model", "length", "pair_group_id"])
    )


def _table_metric_units(
    comparison: pd.DataFrame,
    diffusion: pd.DataFrame,
) -> pd.DataFrame:
    """Join the three table endpoints at the transcript-pair grain."""

    context = _final_context_endpoints(comparison)
    diffusion_final = _final_diffusion_endpoints(diffusion)
    needed = {"model", "length", "pair_group_id", *TABLE_METRIC_COLUMNS}
    if (
        context.empty
        or diffusion_final.empty
        or needed - set(context) - set(diffusion_final)
    ):
        return pd.DataFrame()
    return context.merge(
        diffusion_final[
            ["model", "length", "pair_group_id", "relative_distal_075"]
        ],
        on=["model", "length", "pair_group_id"],
        how="inner",
    )


def _write_table_metric_summary(units: pd.DataFrame, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    if units.empty:
        summary = pd.DataFrame(
            columns=["model", "length", "n_pairs", *TABLE_METRIC_COLUMNS]
        )
    else:
        summary = (
            units.groupby(["model", "length"], as_index=False)
            .agg(
                n_pairs=("pair_group_id", "nunique"),
                **{
                    column: (column, "mean")
                    for column in TABLE_METRIC_COLUMNS
                },
            )
            .copy()
        )
        model_rank = {
            name: index for index, name in enumerate(ordered_models(summary["model"]))
        }
        summary["_model_order"] = summary["model"].map(model_rank)
        summary = summary.sort_values(
            ["length", "_model_order"]
        ).drop(columns="_model_order")
    path = output_dir / "00_final_layer_table_metrics.csv"
    summary.to_csv(path, index=False)
    return str(path)


def _final_diffusion_endpoints(diffusion: pd.DataFrame) -> pd.DataFrame:
    """Return one final-layer diffusion record per paired transcript."""

    metrics = [
        "relative_distal_025",
        "relative_distal_050",
        "relative_distal_075",
    ]
    required = {"model", "length", "pair_group_id", "layer", *metrics}
    if diffusion.empty or required - set(diffusion):
        return pd.DataFrame()
    keys = ["model", "length", "pair_group_id", "layer"]
    available_metadata = [
        column
        for column in ("seed", "relative_distal_threshold")
        if column in diffusion
    ]
    by_layer = (
        diffusion.groupby(keys, as_index=False, dropna=False)[
            [*available_metadata, *metrics]
        ]
        .first()
        .sort_values(keys)
    )
    final_layer = by_layer.groupby(
        ["model", "length", "pair_group_id"]
    )["layer"].transform("max")
    return by_layer[by_layer["layer"] == final_layer].copy()


def _plot_length_scaling_overview(
    comparison: pd.DataFrame,
    diffusion: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
) -> list[str]:
    """Compare final context discrimination with fixed-relative diffusion."""

    context_summary = _final_context_endpoints(comparison)
    diffusion_final = _final_diffusion_endpoints(diffusion)
    if (
        context_summary.empty
        or diffusion_final.empty
        or "delta_context_separation_structured_vs_native" not in context_summary
        or context_summary["length"].nunique() < 2
    ):
        return []
    import matplotlib.pyplot as plt  # type: ignore

    models = _ordered_models(
        set(context_summary["model"]) & set(diffusion_final["model"])
    )
    display_models = [display_model_name(model) for model in models]
    palette = display_model_colors(models)
    context_summary = _with_length_label(_with_display_model(context_summary))
    diffusion_final = _with_length_label(_with_display_model(diffusion_final))
    figure, axes = plt.subplots(
        1, 2, figsize=(15, 6), constrained_layout=True
    )
    specs = (
        (
            axes[0],
            context_summary,
            "delta_context_separation_structured_vs_native",
            "Additional Context Separation from structured rearrangement",
            "Additional Context Separation (higher = stronger)",
        ),
        (
            axes[1],
            diffusion_final,
            "relative_distal_075",
            "Representation diffusion in the outermost 25% of available distance",
            "Relative-distal diffusion, 1 − cosine (lower = more selective)",
        ),
    )
    for axis, data, metric, title, ylabel in specs:
        _draw_length_barplot(
            axis,
            data,
            metric,
            display_models=display_models,
            palette=palette,
        )
        _style_length_bar_axis(
            axis,
            title=title,
            ylabel=ylabel,
            show_legend=True,
        )
    figure.suptitle(
        "Final-layer context discrimination and fixed-relative distal diffusion",
        fontsize=15,
    )
    paths = _save_figure(
        figure,
        output_dir,
        f"{stem_prefix}__05_length_scaling_overview",
    )
    plt.close(figure)
    return paths


def _style_figure7_axis(
    axis: Any,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    show_legend: bool,
    show_xlabel: bool,
) -> None:
    _style_length_bar_axis(
        axis,
        title=title,
        ylabel=ylabel,
        show_legend=show_legend,
    )
    axis.set_ylim(*ylim)
    axis.set_facecolor("white")
    if show_xlabel:
        axis.set_xlabel("Input length (nt)")
    else:
        axis.set_xlabel("")


def _plot_length_scaling_table_metrics(
    comparison: pd.DataFrame,
    diffusion: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
) -> list[str]:
    """Plot the three table endpoints as a stacked length-scaling chart."""

    units = _table_metric_units(comparison, diffusion)
    paths: list[str] = []
    if not units.empty:
        paths.append(_write_table_metric_summary(units, output_dir))
    if units.empty or units["length"].nunique() < 2:
        return paths
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    sns.set_theme(style="whitegrid", context="notebook")
    models = _ordered_models(units["model"])
    display_models = [display_model_name(model) for model in models]
    palette = display_model_colors(models)
    plot_data = _with_length_label(_with_display_model(units))
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 10.8),
        sharex=True,
        facecolor="white",
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    specs = (
        (
            axes[0],
            "delta_context_separation_structured_vs_native",
            "Additional Context Separation",
            "ΔCS ↑",
            (0.0, 0.5),
        ),
        (
            axes[1],
            "cross_context_same_base_cos_structured",
            "Cross-region Same-base Similarity",
            "C_cross ↓",
            (0.0, 0.8),
        ),
        (
            axes[2],
            "relative_distal_075",
            "Relative-distal Diffusion",
            "D_distal",
            (0.0, 0.05),
        ),
    )
    for index, (axis, metric, title, ylabel, ylim) in enumerate(specs):
        _draw_length_barplot(
            axis,
            plot_data,
            metric,
            display_models=display_models,
            palette=palette,
        )
        _style_figure7_axis(
            axis,
            title=title,
            ylabel=ylabel,
            ylim=ylim,
            show_legend=index == 0,
            show_xlabel=index == len(specs) - 1,
        )
        if index < len(specs) - 1:
            axis.tick_params(axis="x", labelbottom=False)
    paths.extend(
        _save_figure(
            figure,
            output_dir,
            f"{stem_prefix}__07_table_metrics_overview",
        )
    )
    plt.close(figure)
    return paths


def _plot_diffusion_threshold_sensitivity(
    diffusion: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
) -> list[str]:
    """Show final-layer sensitivity to common relative-distance cutoffs."""

    final = _final_diffusion_endpoints(diffusion)
    if final.empty:
        return []
    import math

    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    models = _ordered_models(final["model"].dropna().unique())
    display_models = [display_model_name(model) for model in models]
    relative = _with_display_model(final).melt(
        id_vars=["model", "length", "pair_group_id"],
        value_vars=[
            "relative_distal_025",
            "relative_distal_050",
            "relative_distal_075",
        ],
        var_name="metric",
        value_name="diffusion",
    )
    relative["threshold"] = relative["metric"].map(
        {
            "relative_distal_025": 0.25,
            "relative_distal_050": 0.50,
            "relative_distal_075": 0.75,
        }
    )
    relative["threshold_label"] = relative["threshold"].map(
        {
            0.25: "≥ 0.25",
            0.50: "≥ 0.50",
            0.75: "≥ 0.75",
        }
    )
    threshold_palette = {
        "≥ 0.25": "#4C78A8",
        "≥ 0.50": "#F58518",
        "≥ 0.75": "#54A24B",
    }
    threshold_labels = ["≥ 0.25", "≥ 0.50", "≥ 0.75"]
    lengths = sorted(relative["length"].dropna().unique())
    length_labels = [str(int(value)) for value in lengths]
    relative["length_label"] = relative["length"].map(
        lambda value: str(int(value))
    )
    columns = min(4, len(display_models))
    rows = max(1, math.ceil(len(display_models) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 3.8 * rows),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for index, model in enumerate(display_models):
        row = index // columns
        column = index % columns
        axis = axes[row, column]
        model_data = relative[relative["model"] == model]
        sns.lineplot(
            data=model_data,
            x="length_label",
            y="diffusion",
            hue="threshold_label",
            hue_order=threshold_labels,
            palette=threshold_palette,
            estimator="mean",
            errorbar=("ci", 95),
            n_boot=2000,
            seed=20260709,
            marker="o",
            linewidth=2.0,
            ax=axis,
        )
        axis.set_title(model, fontsize=10)
        axis.set_xlabel("Input length (nt)")
        if column == 0:
            axis.set_ylabel("Relative distal diffusion")
        else:
            axis.set_ylabel("")
        axis.set_xticks(range(len(length_labels)))
        axis.set_xticklabels(length_labels, rotation=30, ha="right")
        axis.grid(True, alpha=0.25)
        legend = axis.get_legend()
        if index == 0:
            axis.legend(
                title="Distance cutoff",
                fontsize=8,
                title_fontsize=8,
                frameon=True,
                framealpha=0.85,
            )
        elif legend is not None:
            legend.remove()
    for axis in list(axes.flat)[len(display_models) :]:
        axis.axis("off")
    figure.suptitle(
        "Final-layer diffusion sensitivity to fixed relative-distance threshold",
        fontsize=15,
    )
    paths = _save_figure(
        figure,
        output_dir,
        f"{stem_prefix}__06_diffusion_threshold_sensitivity",
    )
    plt.close(figure)
    return paths


def plot_cosine_diagnostics(
    context: pd.DataFrame,
    diffusion: pd.DataFrame,
    output_dir: Path,
    *,
    context_controls: pd.DataFrame | None = None,
    trajectories: pd.DataFrame | None = None,
    attention_similarity: pd.DataFrame | None = None,
) -> list[str]:
    """Generate complementary cosine plots and machine-readable summaries."""

    if context.empty or diffusion.empty:
        return []
    import seaborn as sns  # type: ignore

    sns.set_theme(style="whitegrid", context="notebook")
    all_context = _prepare_context(context, structured_only=False)
    prepared_context = _prepare_context(context, structured_only=True)
    prepared_diffusion = _prepare_diffusion(diffusion)
    if prepared_context.empty or prepared_diffusion.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in CLEANUP_EXTENSIONS:
        for path in output_dir.glob(f"*__0[1-7]_*.{extension}"):
            path.unlink()
    outputs = _write_summaries(
        prepared_context, prepared_diffusion, output_dir
    )
    outputs.append(_write_control_summary(all_context, output_dir))
    control_frame = (
        context_controls
        if context_controls is not None
        else pd.DataFrame()
    )
    trajectory_frame = (
        trajectories if trajectories is not None else pd.DataFrame()
    )
    association_frame = (
        attention_similarity
        if attention_similarity is not None
        else pd.DataFrame()
    )
    lengths = sorted(
        set(prepared_context["length"].dropna())
        & set(prepared_diffusion["length"].dropna())
    )
    models = _ordered_models(prepared_context["model"].dropna().unique())
    length_text = ", ".join(str(int(value)) for value in lengths)
    title = (
        f"Lengths {length_text} nt | "
        + " vs ".join(display_model_name(model) for model in models)
    )
    prefix = (
        f"lengths{int(min(lengths))}-{int(max(lengths))}"
        if lengths
        else "lengths"
    )
    outputs.extend(
        _plot_context_geometry(
            control_frame,
            output_dir,
            prefix,
            title,
        )
    )
    outputs.extend(
        _plot_joint_trajectory(
            prepared_context,
            prepared_diffusion,
            output_dir,
            prefix,
            title,
        )
    )
    outputs.extend(
        _plot_centroid_margins(
            trajectory_frame,
            output_dir,
            prefix,
            title,
        )
    )
    outputs.extend(
        _plot_attention_similarity_coupling(
            association_frame,
            output_dir,
            prefix,
        )
    )
    outputs.extend(
        _plot_length_scaling_overview(
            control_frame,
            prepared_diffusion,
            output_dir,
            prefix,
        )
    )
    outputs.extend(
        _plot_diffusion_threshold_sensitivity(
            prepared_diffusion,
            output_dir,
            prefix,
        )
    )
    outputs.extend(
        _plot_length_scaling_table_metrics(
            control_frame,
            prepared_diffusion,
            output_dir,
            prefix,
        )
    )
    return outputs
