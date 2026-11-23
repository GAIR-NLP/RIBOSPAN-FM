# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-sequence score figure of a secondary-structure benchmark.

Top row: one violin per model for each test set scored as a whole, with the
median as a line, the mean as a dot, and one of the two as a number above.
Bottom row: one box per model for each RNA family, in the order and colors of
the violins. There is no legend: the violins name the models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# The RIBOSPAN colors are those of the long-context benchmark's charts.
MODEL_COLORS = {
    "LIMBER-macro": "#8F7BA3",
    "structRFM": "#7B9BB8",
    "RIBOSPAN-1K-15": "#596C9B",
    "RIBOSPAN-1K-40": "#557F73",
    "RIBOSPAN-10K-15": "#C2A454",
    "RIBOSPAN-10K-40": "#B87558",
}
FALLBACK_COLOR = "#77736D"
FAMILY_LABELS = {
    "5s": "5S",
    "16s": "16S",
    "23s": "23S",
    "tRNA": "tRNA",
    "tmRNA": "tmRNA",
    "srp": "SRP",
    "RNaseP": "RNase P",
    "grp1": "Group I",
    "grp2": "Group II",
    "telomerase": "Telomerase",
}
VECTOR_EXTENSIONS = ("pdf", "svg")
# Inches. The width is for three models and grows with more.
FIGURE_WIDTH = 8.0
FIGURE_WIDTH_MODELS = 3
ROW_HEIGHT = 4.38
MARK = "#444444"
MARK_WIDTH = 1.1
# Model and family names under the panels, sized so neighbors do not touch.
TICK_LABEL_FONTSIZE = 10
VIOLIN_WIDTH = 0.8
BOX_GROUP_WIDTH = 0.8
# Room above 1.0 for the number over each violin.
Y_TOP = 1.12
NUMBER_HEIGHT = 1.03


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, FALLBACK_COLOR)


def _model_label(model: str) -> str:
    """Break a RIBOSPAN name after the family so neighbors do not touch."""

    family, separator, variant = model.partition("-")
    return f"{family}\n{variant}" if family == "RIBOSPAN" and separator else model


def _style_axis(axis: Any, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("")
    axis.set_facecolor("white")
    axis.grid(True, axis="y", alpha=0.25)
    axis.grid(False, axis="x")
    axis.set_ylim(0.0, Y_TOP)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))


def _half_width_at(body: Any, center: float, height: float) -> float:
    """Half the width of a violin body at ``height``."""

    vertices = body.get_paths()[0].vertices
    right = vertices[vertices[:, 0] > center]
    order = np.argsort(right[:, 1])
    return float(np.interp(height, right[order, 1], right[order, 0]) - center)


def _draw_violins(axis: Any, scores: Mapping[str, np.ndarray], number: str) -> None:
    for index, (model, values) in enumerate(scores.items()):
        median, mean = float(np.median(values)), float(np.mean(values))
        reach = VIOLIN_WIDTH / 2
        # A density needs at least two distinct scores.
        if np.ptp(values) > 0:
            body = axis.violinplot(
                [values],
                positions=[index],
                widths=VIOLIN_WIDTH,
                showextrema=False,
                points=400,
            )["bodies"][0]
            # A white edge, as on the boxes, would erase the thin tails.
            body.set_facecolor(_color(model))
            body.set_edgecolor(_color(model))
            body.set_linewidth(0.8)
            body.set_alpha(1.0)
            reach = _half_width_at(body, index, median)
        axis.plot(
            [index - reach, index + reach],
            [median, median],
            color=MARK,
            linewidth=MARK_WIDTH + 0.4,
            solid_capstyle="butt",
            zorder=3,
        )
        axis.plot(
            [index],
            [mean],
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=MARK,
            markeredgewidth=MARK_WIDTH,
            zorder=4,
        )
        axis.text(
            index,
            NUMBER_HEIGHT,
            f"{median if number == 'median' else mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_xticks(range(len(scores)))
    axis.set_xticklabels(
        [_model_label(model) for model in scores], fontsize=TICK_LABEL_FONTSIZE
    )
    axis.set_xlim(-0.6, len(scores) - 0.4)


def _draw_family_boxes(
    axis: Any, by_family: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    """One box per model inside each family group; whiskers reach 1.5 IQR."""

    labels = []
    for group, (family, scores) in enumerate(by_family.items()):
        width = BOX_GROUP_WIDTH / len(scores)
        for index, (model, values) in enumerate(scores.items()):
            axis.boxplot(
                [values],
                positions=[group - BOX_GROUP_WIDTH / 2 + width * (index + 0.5)],
                widths=width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": _color(model),
                    "edgecolor": "white",
                    "linewidth": 1.0,
                },
                medianprops={"color": MARK, "linewidth": MARK_WIDTH + 0.4},
                whiskerprops={"color": MARK, "linewidth": MARK_WIDTH},
                capprops={"color": MARK, "linewidth": MARK_WIDTH},
                capwidths=width * 0.45,
            )
        count = len(next(iter(scores.values())))
        labels.append(f"{FAMILY_LABELS.get(family, family)}\n(n = {count:,})")
    axis.set_xticks(range(len(by_family)))
    axis.set_xticklabels(labels, fontsize=TICK_LABEL_FONTSIZE)
    axis.set_xlim(-0.6, len(by_family) - 0.4)


def plot_score_distributions(
    whole_sets: Sequence[tuple[str, Mapping[str, Sequence[float]]]],
    family_title: str,
    by_family: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    ylabel: str,
    number: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    """Write the figure as PDF and SVG and return the paths.

    ``whole_sets`` holds a title and each model's per-sequence scores for every
    test set of the top row; ``by_family`` holds the same per RNA family.
    ``number`` is ``"median"`` or ``"mean"``, the statistic written above each
    violin. Every mapping lists the models in the order they are drawn.
    """

    if number not in {"median", "mean"}:
        raise ValueError(f"number must be 'median' or 'mean', got {number!r}")
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    def arrays(scores: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
        return {model: np.asarray(values, dtype=float) for model, values in scores.items()}

    sns.set_theme(style="whitegrid", context="notebook")
    # Text stays text in Illustrator.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    models = max(len(scores) for _, scores in whole_sets)
    figure = plt.figure(
        figsize=(
            FIGURE_WIDTH * max(1.0, models / FIGURE_WIDTH_MODELS),
            ROW_HEIGHT * 2,
        ),
        facecolor="white",
        constrained_layout=True,
    )
    grid = figure.add_gridspec(2, len(whole_sets))
    for column, (title, scores) in enumerate(whole_sets):
        axis = figure.add_subplot(grid[0, column])
        scores = arrays(scores)
        _draw_violins(axis, scores, number)
        count = len(next(iter(scores.values())))
        _style_axis(axis, f"{title} (n = {count:,})", ylabel)
    axis = figure.add_subplot(grid[1, :])
    _draw_family_boxes(axis, {family: arrays(scores) for family, scores in by_family.items()})
    _style_axis(axis, family_title, ylabel)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in VECTOR_EXTENSIONS:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
        paths.append(path)
    plt.close(figure)
    return paths
