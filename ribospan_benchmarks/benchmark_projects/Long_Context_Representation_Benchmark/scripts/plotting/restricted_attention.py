# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Publication plot for attention held to the pretraining span."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cosine import (
    LENGTH_BARPLOT_KWARGS,
    MLM_ACCURACY_SPEC,
    TABLE_FIGURE_WIDTH,
    TABLE_METRIC_SPECS,
    TABLE_ROW_HEIGHT,
    _draw_length_barplot,
    _style_figure7_axis,
    _with_display_model,
    _with_length_label,
)
from .export import CLEANUP_EXTENSIONS, save_figure
from .model_style import (
    display_model_colors,
    display_model_name,
    model_color,
    ordered_models,
)


INCREASE = "Increase when restricted"
DECREASE = "Decrease when restricted"
HATCH = "////"
LEGEND_HATCH = "//////"
HATCH_LINEWIDTH = 0.6
# White line across a bar at its unrestricted level, in points.
START_LINE = 1.0
# Visible thickness of the frame line, in points.
FRAME = 0.9
# Share of white mixed into the model's base color for the frame line.
FRAME_LIGHTEN = 0.6
# A change below this share of the axis range cannot be seen; the bar is left bare.
MIN_CHANGE = 0.01
NEUTRAL = "#8A8A8A"
# The unrestricted interval is white where it lies on the bar, which is when
# restriction raised the value, and light gray where it lies on the background.
INTERVAL_ON_BAR = "white"
INTERVAL_ON_BACKGROUND = "#B8B8B8"
INTERVAL_LINEWIDTH = 1.0
# Half the cap width, as a share of the bar width.
INTERVAL_CAP = 0.22


def _lighter(color: Any) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb  # type: ignore

    red, green, blue = to_rgb(color)
    return (
        red + (1.0 - red) * FRAME_LIGHTEN,
        green + (1.0 - green) * FRAME_LIGHTEN,
        blue + (1.0 - blue) * FRAME_LIGHTEN,
    )


def _hatch_segment(axis: Any, patch: Any, bottom: float, top: float) -> None:
    """White hatch lines over a bar between two heights, and a line at the lower one."""

    from matplotlib.patches import Rectangle  # type: ignore

    segment = Rectangle(
        (patch.get_x(), bottom),
        patch.get_width(),
        top - bottom,
        facecolor="none",
        edgecolor="none",
        linewidth=0,
        hatch=HATCH,
        zorder=patch.get_zorder() + 0.1,
    )
    segment.set_hatch_linewidth(HATCH_LINEWIDTH)
    segment.set_hatchcolor("white")
    axis.add_patch(segment)
    axis.plot(
        [patch.get_x(), patch.get_x() + patch.get_width()],
        [bottom, bottom],
        color="white",
        linewidth=START_LINE,
        solid_capstyle="butt",
        zorder=patch.get_zorder() + 0.12,
    )


def _frame_segment(
    axis: Any, patch: Any, bottom: float, top: float, color: Any
) -> None:
    """An empty frame, open at the bottom, as wide as the colored part of a bar.

    The colored line is the inner half of a stroke on the bar's own outline,
    under the same white edge the bars have, so the frame is flush with the bar.
    """

    from matplotlib.patches import PathPatch, Rectangle  # type: ignore
    from matplotlib.path import Path as ArtistPath  # type: ignore

    left, width = patch.get_x(), patch.get_width()
    outline = ArtistPath(
        [(left, bottom), (left, top), (left + width, top), (left + width, bottom)]
    )
    edge = patch.get_linewidth()
    color_line = PathPatch(
        outline,
        facecolor="none",
        edgecolor=color,
        linewidth=edge + 2 * FRAME,
        capstyle="butt",
        joinstyle="miter",
        zorder=patch.get_zorder() + 0.1,
    )
    axis.add_patch(color_line)
    # The clip is set after add_patch, which would replace it with the axes box.
    color_line.set_clip_path(
        Rectangle((left, bottom), width, top - bottom, transform=axis.transData)
    )
    axis.add_patch(
        PathPatch(
            outline,
            facecolor="none",
            edgecolor=patch.get_edgecolor(),
            linewidth=edge,
            capstyle="butt",
            joinstyle="miter",
            zorder=patch.get_zorder() + 0.11,
        )
    )


def _unrestricted_intervals(
    reference: pd.DataFrame, metric: str
) -> dict[tuple[str, str], tuple[float, float]]:
    """The interval the bar chart would draw for each unrestricted mean.

    Uses the bars' own bootstrap settings, so the intervals equal those of the
    unrestricted table-metric chart.
    """

    import seaborn as sns  # type: ignore

    _, level = LENGTH_BARPLOT_KWARGS["errorbar"]
    tail = (100.0 - level) / 2.0
    intervals: dict[tuple[str, str], tuple[float, float]] = {}
    for key, values in reference.groupby(["model", "length_label"])[metric]:
        values = values.dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue
        draws = sns.algorithms.bootstrap(
            values,
            func=LENGTH_BARPLOT_KWARGS["estimator"],
            n_boot=LENGTH_BARPLOT_KWARGS["n_boot"],
            seed=LENGTH_BARPLOT_KWARGS["seed"],
        )
        low, high = np.percentile(draws, [tail, 100.0 - tail])
        intervals[key] = (float(low), float(high))
    return intervals


def _draw_interval(axis: Any, patch: Any, low: float, high: float, color: Any) -> None:
    """A capped interval at the center of a bar, above the change mark."""

    center = patch.get_x() + patch.get_width() / 2
    cap = patch.get_width() * INTERVAL_CAP
    style = {
        "color": color,
        "linewidth": INTERVAL_LINEWIDTH,
        "solid_capstyle": "butt",
        "zorder": patch.get_zorder() + 0.13,
    }
    axis.plot([center, center], [low, high], **style)
    for height in (low, high):
        axis.plot([center - cap, center + cap], [height, height], **style)


def _mark_changes(
    axis: Any,
    plot_data: pd.DataFrame,
    unrestricted_means: dict[tuple[str, str], float],
    unrestricted_intervals: dict[tuple[str, str], tuple[float, float]],
    metric: str,
    display_models: list[str],
    frame_colors: dict[str, Any],
    ylim: tuple[float, float],
) -> None:
    """Mark on each bar the segment between its unrestricted and restricted means.

    A marked bar also gets the interval of its unrestricted mean.
    """

    drawn = [
        model
        for model in display_models
        if plot_data.loc[plot_data["model"] == model, metric].notna().any()
    ]
    if len(axis.containers) != len(drawn):
        raise RuntimeError(
            f"{metric}: {len(axis.containers)} bar groups for {len(drawn)} models"
        )
    # seaborn places the length groups in their order of appearance.
    labels = list(pd.unique(plot_data["length_label"]))
    threshold = MIN_CHANGE * (ylim[1] - ylim[0])
    for container, model in zip(axis.containers, drawn):
        for patch in container:
            group = int(round(patch.get_x() + patch.get_width() / 2))
            level = unrestricted_means.get((model, labels[group]))
            top = patch.get_height()
            if level is None or pd.isna(level) or pd.isna(top):
                continue
            if abs(level - top) < threshold:
                continue
            if top > level:
                _hatch_segment(axis, patch, level, top)
            else:
                _frame_segment(axis, patch, top, level, frame_colors[model])
            interval = unrestricted_intervals.get((model, labels[group]))
            if interval is not None:
                _draw_interval(
                    axis,
                    patch,
                    *interval,
                    INTERVAL_ON_BAR if top > level else INTERVAL_ON_BACKGROUND,
                )


def _change_handles() -> list[Any]:
    from matplotlib.patches import Patch  # type: ignore

    increase = Patch(facecolor=NEUTRAL, edgecolor="white", label=INCREASE)
    increase.set_hatch(LEGEND_HATCH)
    increase.set_hatch_linewidth(HATCH_LINEWIDTH)
    increase.set_hatchcolor("white")
    decrease = Patch(
        facecolor="white",
        edgecolor=_lighter(NEUTRAL),
        linewidth=FRAME,
        label=DECREASE,
    )
    return [increase, decrease]


def plot_restricted_attention(
    unrestricted: pd.DataFrame,
    restricted: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Redraw the table-metric chart with attention held to the pretraining span.

    Both frames hold one row per model, length, and transcript pair. Bars show
    the restricted values. On each bar the segment between the unrestricted and
    the restricted mean is marked: white hatch lines where restriction raised
    the value, an empty frame above the bar where it lowered the value. The
    unrestricted end of the mark carries the interval of the unrestricted mean.
    """

    keys = {"model", "length", "pair_group_id"}
    if (
        unrestricted.empty
        or restricted.empty
        or keys - set(unrestricted)
        or keys - set(restricted)
    ):
        return []
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    units = restricted.copy()
    units["length"] = pd.to_numeric(units["length"], errors="coerce")
    units = units.dropna(subset=["length"]).sort_values("length", kind="stable")
    specs = [
        spec
        for spec in (*TABLE_METRIC_SPECS, MLM_ACCURACY_SPEC)
        if spec[0] in units and units[spec[0]].notna().any()
    ]
    if units.empty or not specs:
        return []
    stem = (
        f"lengths{int(units['length'].min())}-{int(units['length'].max())}"
        "__08_restricted_attention"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in CLEANUP_EXTENSIONS:
        for path in output_dir.glob(f"*__08_*.{extension}"):
            path.unlink()

    sns.set_theme(style="whitegrid", context="notebook")
    models = ordered_models(units["model"])
    display_models = [display_model_name(model) for model in models]
    palette = display_model_colors(models)
    # A model drawn in the pale shade next to its YaRN variant takes the frame
    # from its base color, or the frame would fade into the background.
    frame_colors = {
        name: _lighter(model_color(model) or palette[name])
        for model, name in zip(models, display_models)
    }
    plot_data = _with_length_label(_with_display_model(units))
    reference = unrestricted[unrestricted["model"].isin(set(units["model"]))].copy()
    reference["length"] = pd.to_numeric(reference["length"], errors="coerce")
    reference = _with_length_label(
        _with_display_model(reference.dropna(subset=["length"]))
    )
    figure, axes = plt.subplots(
        len(specs),
        1,
        figsize=(TABLE_FIGURE_WIDTH, TABLE_ROW_HEIGHT * len(specs)),
        sharex=True,
        facecolor="white",
        constrained_layout=True,
        squeeze=False,
    )
    figure.patch.set_facecolor("white")
    change_handles = _change_handles()
    for index, (axis, (metric, title, ylabel, ylim)) in enumerate(
        zip(axes[:, 0], specs)
    ):
        _draw_length_barplot(
            axis,
            plot_data,
            metric,
            display_models=display_models,
            palette=palette,
        )
        unrestricted_means: dict[tuple[str, str], float] = {}
        unrestricted_intervals: dict[tuple[str, str], tuple[float, float]] = {}
        if metric in reference:
            unrestricted_means = (
                reference.groupby(["model", "length_label"])[metric].mean().to_dict()
            )
            unrestricted_intervals = _unrestricted_intervals(reference, metric)
        _mark_changes(
            axis,
            plot_data,
            unrestricted_means,
            unrestricted_intervals,
            metric,
            display_models,
            frame_colors,
            ylim,
        )
        _style_figure7_axis(
            axis,
            title=title,
            ylabel=ylabel,
            ylim=ylim,
            show_legend=index == 0,
            show_xlabel=index == len(specs) - 1,
            legend_extra_handles=change_handles,
        )
        if index < len(specs) - 1:
            axis.tick_params(axis="x", labelbottom=False)
    paths = save_figure(figure, output_dir, stem)
    plt.close(figure)
    return paths
