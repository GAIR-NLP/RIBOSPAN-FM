# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared atlas / 2x2 panel drawing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.analyze import (
    PLOT_EXCLUDE_TYPES,
    RNA_TYPE_COLORS,
    color_for_rna_type,
    ordered_rna_types,
)
from scripts.plotting.export import EXPORT_FORMATS, save_figure

# POINT_SIZE matches the standalone atlas; composite scales by axes width.
POINT_SIZE = 2.5
POINT_ALPHA = 0.55
REF_AXES_WIDTH_INCH = 8.73


def axes_width_inch(ax) -> float:
    pos = ax.get_position()
    return float(pos.width) * float(ax.figure.get_figwidth())


def marker_size_scale_for_ax(ax, *, ref_width_inch: float = REF_AXES_WIDTH_INCH) -> float:
    """Scale scatter ``s`` so marker diameter tracks axes width (area ∝ width²)."""
    width = axes_width_inch(ax)
    if width <= 0 or ref_width_inch <= 0:
        return 1.0
    return (width / float(ref_width_inch)) ** 2


def resolve_size_scale(ax, size_scale: float | None) -> float:
    """``None`` → auto from axes size; otherwise use the explicit multiplier."""
    if size_scale is None:
        return marker_size_scale_for_ax(ax)
    return float(size_scale)

HOUSEKEEPING_TYPES: frozenset[str] = frozenset({"rRNA", "tRNA", "tmRNA"})
REGULATORY_TYPES: frozenset[str] = frozenset(
    {
        "lncRNA",
        "snoRNA",
        "miRNA",
        "pre_miRNA",
        "siRNA",
        "snRNA",
        "piRNA",
    }
)
CODING_TYPES: frozenset[str] = frozenset({"mRNA"})

FUNCTIONAL_CLASS_COLORS = {
    "housekeeping": "#1f77b4",
    "regulatory": "#d62728",
    "coding": "#2ca02c",
}

REGULATORY_PANEL_COLORS: dict[str, str] = {
    "lncRNA": "#33A02C",
    "snRNA": "#6A3D9A",
    "snoRNA": "#1F78B4",
    "miRNA": "#E31A1C",
    "pre_miRNA": "#FF7F00",
    "piRNA": "#E7298A",
    "siRNA": "#B15928",
}

# High-contrast long-RNA palette (atlas greens/browns collapse several types).
LONG_PANEL_COLORS: dict[str, str] = {
    "mRNA": "#0072B2",
    "lncRNA": "#D55E00",
    "rRNA": "#009E73",
    "misc_RNA": "#CC79A7",
    "sRNA": "#E69F00",
    "ncRNA": "#56B4E9",
    "antisense_RNA": "#882255",
    "other": "#332288",
}

FAMILY_PALETTE: list[str] = list(
    dict.fromkeys(
        [
            "#0077BB",
            "#EE7733",
            "#009988",
            *REGULATORY_PANEL_COLORS.values(),
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
            "#A6761D",
            "#D95F02",
            "#7570B3",
            "#1B9E77",
            "#E7298A",
            "#66A61E",
            "#E6AB02",
            "#666666",
            "#1F78B4",
            "#B2DF8A",
            "#222222",
        ]
    )
)


def point_style(_n: int = 0) -> tuple[float, float]:
    """Uniform marker size/alpha."""
    return POINT_SIZE, POINT_ALPHA


def rfam_color_map(label_order: list[str]) -> dict[str, str]:
    return {
        lab: FAMILY_PALETTE[i % len(FAMILY_PALETTE)]
        for i, lab in enumerate(label_order)
    }


def _scatter_by_labels(
    ax,
    coords: np.ndarray,
    labels: list[str],
    *,
    draw_types: Iterable[str] | None = None,
    color_map: dict[str, str] | None = None,
    size_scale: float | None = 1.0,
    rasterized: bool = False,
) -> list[tuple[str, str]]:
    """Scatter points colored by label. Returns legend (name, color) in draw order.

    ``size_scale=None`` auto-scales markers to the axes width (composite); standalone
    plots pass ``1.0`` to keep the calibrated POINT_SIZE.
    """
    types = ordered_rna_types(labels) if draw_types is None else list(draw_types)
    present = {lab for lab in labels}
    types = [t for t in types if t in present]
    counts = {t: sum(1 for lab in labels if lab == t) for t in types}
    draw_order = sorted(types, key=lambda t: counts[t], reverse=True)

    fallback_i = 0
    legend_items: list[tuple[str, str]] = []
    seen: set[str] = set()
    size, alpha = point_style()
    scale = resolve_size_scale(ax, size_scale)

    for rna_type in draw_order:
        mask = np.array([lab == rna_type for lab in labels], dtype=bool)
        if not mask.any():
            continue
        if color_map is not None and rna_type in color_map:
            color = color_map[rna_type]
        else:
            color = color_for_rna_type(rna_type, fallback_i)
            if rna_type not in RNA_TYPE_COLORS:
                fallback_i += 1
        n = int(counts[rna_type])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size * scale,
            alpha=alpha,
            color=color,
            linewidths=0,
            rasterized=rasterized,
            zorder=2 if n < 500 else 1,
        )
        if rna_type not in seen:
            legend_items.append((rna_type, color))
            seen.add(rna_type)

    ordered_legend = [(t, c) for t, c in ((t, dict(legend_items).get(t)) for t in types) if c]
    return ordered_legend


def _square_pad_limits(xlim, ylim) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pad limits so x/y spans match; keeps equal-aspect boxes the same size."""
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0, 1e-6)
    return (cx - half, cx + half), (cy - half, cy + half)


def _style_panel(ax, *, title: str, xlim, ylim, title_fontsize: float = 11) -> None:
    xlim, ylim = _square_pad_limits(xlim, ylim)
    ax.set_title(title, fontsize=title_fontsize, pad=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Keep square axes; avoid shrinking when t-SNE x/y spans differ.
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.9)


def _add_legend(
    ax,
    items: list[tuple[str, str]],
    *,
    fontsize: float = 7.5,
    ncol: int = 1,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=4.5,
            label=name,
        )
        for name, color in items
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.22,
        handletextpad=0.3,
        columnspacing=0.8,
        fontsize=fontsize,
        ncol=max(1, int(ncol)),
    )


def _prepare_coords_labels(
    coords: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, list[str], tuple[float, float], tuple[float, float]]:
    keep = np.array([lab not in PLOT_EXCLUDE_TYPES for lab in labels], dtype=bool)
    coords = np.asarray(coords)[keep]
    labels = [lab for lab, ok in zip(labels, keep) if ok]
    pad = 3.0
    xlim = (float(coords[:, 0].min()) - pad, float(coords[:, 0].max()) + pad)
    ylim = (float(coords[:, 1].min()) - pad, float(coords[:, 1].max()) + pad)
    return coords, labels, xlim, ylim


def draw_full_atlas_on_ax(
    ax,
    coords: np.ndarray,
    labels: list[str],
    *,
    title: str,
    xlim=None,
    ylim=None,
    size_scale: float | None = 1.0,
    rasterized: bool = False,
    show_legend: bool = False,
    title_fontsize: float = 11,
) -> None:
    """All-type atlas on one axes (shared style with single-model plots)."""
    coords, labels, auto_xlim, auto_ylim = _prepare_coords_labels(coords, labels)
    if xlim is None:
        xlim = auto_xlim
    if ylim is None:
        ylim = auto_ylim
    legend = _scatter_by_labels(
        ax,
        coords,
        labels,
        draw_types=ordered_rna_types(labels),
        size_scale=size_scale,
        rasterized=rasterized,
    )
    _style_panel(ax, title=title, xlim=xlim, ylim=ylim, title_fontsize=title_fontsize)
    if show_legend:
        _add_legend(ax, legend, fontsize=5.5)


PANEL_TITLES = {
    "functional": "(1) Functional classes",
    "regulatory": "(2) Regulatory biotypes",
    "long": "(3) Long RNAs",
    "rfam": "(4) Rfam families",
}


def draw_long_on_ax(
    ax,
    coords: np.ndarray,
    labels: list[str],
    label_order: list[str],
    *,
    title: str | None = None,
    threshold: int = 1024,
    size_scale: float | None = 1.0,
    rasterized: bool = False,
    show_legend: bool = False,
    title_fontsize: float = 9,
    legend_fontsize: float = 5.5,
) -> list[tuple[str, str]]:
    """Draw long-sequence biotype scatter (own t-SNE frame)."""
    coords = np.asarray(coords)
    panel_title = title if title is not None else f"(3) Long RNAs (>{threshold} nt)"
    if coords.size == 0:
        _style_panel(
            ax, title=panel_title, xlim=(-1, 1), ylim=(-1, 1), title_fontsize=title_fontsize
        )
        return []

    color_map = {
        lab: LONG_PANEL_COLORS.get(lab, color_for_rna_type(lab, i))
        for i, lab in enumerate(label_order)
    }
    legend = _scatter_by_labels(
        ax,
        coords,
        labels,
        draw_types=label_order,
        color_map=color_map,
        size_scale=size_scale,
        rasterized=rasterized,
    )
    pad = 3.0
    xlim = (float(coords[:, 0].min()) - pad, float(coords[:, 0].max()) + pad)
    ylim = (float(coords[:, 1].min()) - pad, float(coords[:, 1].max()) + pad)
    _style_panel(ax, title=panel_title, xlim=xlim, ylim=ylim, title_fontsize=title_fontsize)
    if show_legend:
        _add_legend(ax, legend, fontsize=legend_fontsize)
    return legend


def draw_rfam_on_ax(
    ax,
    coords: np.ndarray,
    labels: list[str],
    label_order: list[str],
    *,
    title: str = "(4) Rfam families",
    size_scale: float | None = 1.0,
    rasterized: bool = False,
    show_legend: bool = False,
    title_fontsize: float = 9,
    legend_fontsize: float = 5.5,
) -> list[tuple[str, str]]:
    """Draw Rfam-family scatter on one axes (own frame; not shared with biotype panels)."""
    coords = np.asarray(coords)
    if coords.size == 0:
        _style_panel(ax, title=title, xlim=(-1, 1), ylim=(-1, 1), title_fontsize=title_fontsize)
        return []

    counts = {lab: sum(1 for x in labels if x == lab) for lab in label_order}
    draw_order = sorted(label_order, key=lambda t: counts.get(t, 0), reverse=True)
    color_map = rfam_color_map(label_order)
    size, alpha = point_style()
    scale = resolve_size_scale(ax, size_scale)

    for lab in draw_order:
        mask = np.array([x == lab for x in labels], dtype=bool)
        if not mask.any():
            continue
        n = int(counts[lab])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size * scale,
            alpha=alpha,
            color=color_map[lab],
            linewidths=0,
            rasterized=rasterized,
            zorder=2 if n < 500 else 1,
        )

    pad = 3.0
    xlim = (float(coords[:, 0].min()) - pad, float(coords[:, 0].max()) + pad)
    ylim = (float(coords[:, 1].min()) - pad, float(coords[:, 1].max()) + pad)
    _style_panel(ax, title=title, xlim=xlim, ylim=ylim, title_fontsize=title_fontsize)
    legend = [(lab, color_map[lab]) for lab in label_order]
    if show_legend:
        _add_legend(ax, legend, fontsize=legend_fontsize)
    return legend


def draw_functional_panel_on_ax(
    ax,
    panel: str,
    coords: np.ndarray,
    labels: list[str],
    *,
    xlim=None,
    ylim=None,
    size_scale: float | None = 1.0,
    rasterized: bool = False,
    show_legend: bool = False,
    title: str | None = None,
    title_fontsize: float = 9,
    legend_fontsize: float = 5.5,
) -> list[tuple[str, str]]:
    """Draw one shared-frame panel: functional classes | regulatory biotypes."""
    coords, labels, auto_xlim, auto_ylim = _prepare_coords_labels(coords, labels)
    if xlim is None:
        xlim = auto_xlim
    if ylim is None:
        ylim = auto_ylim

    labels_arr = np.array(labels, dtype=object)
    hk_mask = np.array([lab in HOUSEKEEPING_TYPES for lab in labels], dtype=bool)
    reg_mask = np.array([lab in REGULATORY_TYPES for lab in labels], dtype=bool)
    coding_mask = np.array([lab in CODING_TYPES for lab in labels], dtype=bool)

    if panel == "functional":
        class_labels: list[str] = []
        class_coords: list[np.ndarray] = []
        for is_hk, is_reg, is_coding, xy in zip(hk_mask, reg_mask, coding_mask, coords):
            if is_hk:
                class_labels.append("housekeeping")
                class_coords.append(xy)
            elif is_reg:
                class_labels.append("regulatory")
                class_coords.append(xy)
            elif is_coding:
                class_labels.append("coding")
                class_coords.append(xy)
        coords_p = np.asarray(class_coords) if class_coords else np.zeros((0, 2))
        legend = _scatter_by_labels(
            ax,
            coords_p,
            class_labels,
            draw_types=["housekeeping", "regulatory", "coding"],
            color_map=FUNCTIONAL_CLASS_COLORS,
            size_scale=size_scale,
            rasterized=rasterized,
        )
    elif panel == "regulatory":
        legend = _scatter_by_labels(
            ax,
            coords[reg_mask],
            labels_arr[reg_mask].tolist(),
            draw_types=[
                "lncRNA",
                "snRNA",
                "snoRNA",
                "miRNA",
                "pre_miRNA",
                "piRNA",
                "siRNA",
            ],
            color_map=REGULATORY_PANEL_COLORS,
            size_scale=size_scale,
            rasterized=rasterized,
        )
    else:
        raise ValueError(f"unknown shared-frame panel: {panel}")

    panel_title = PANEL_TITLES[panel] if title is None else title
    _style_panel(ax, title=panel_title, xlim=xlim, ylim=ylim, title_fontsize=title_fontsize)
    if show_legend:
        _add_legend(ax, legend, fontsize=legend_fontsize)
    return legend


def draw_functional_quad_on_axes(
    axes_2x2,
    coords: np.ndarray,
    labels: list[str],
    *,
    xlim=None,
    ylim=None,
    size_scale: float | None = 1.0,
    rasterized: bool = False,
    show_legend: bool = True,
    title_fontsize: float = 9,
    legend_fontsize: float = 5.5,
    long_coords: np.ndarray | None = None,
    long_labels: list[str] | None = None,
    long_order: list[str] | None = None,
    long_threshold: int = 1024,
    rfam_coords: np.ndarray | None = None,
    rfam_labels: list[str] | None = None,
    rfam_order: list[str] | None = None,
) -> None:
    """Draw the 4 panels onto a 2x2 axes grid.

    Panel (3) uses dedicated long-seq t-SNE when ``long_coords`` is provided.
    Panel (4) uses Rfam family coloring when ``rfam_coords`` is provided.
    """
    coords, labels, auto_xlim, auto_ylim = _prepare_coords_labels(coords, labels)
    if xlim is None:
        xlim = auto_xlim
    if ylim is None:
        ylim = auto_ylim

    panel_order = (
        ("functional", axes_2x2[0, 0]),
        ("regulatory", axes_2x2[0, 1]),
        ("long", axes_2x2[1, 0]),
        ("rfam", axes_2x2[1, 1]),
    )
    for panel, ax in panel_order:
        if (
            panel == "long"
            and long_coords is not None
            and long_labels is not None
            and long_order is not None
        ):
            draw_long_on_ax(
                ax,
                long_coords,
                long_labels,
                long_order,
                title=f"(3) Long RNAs (>{long_threshold} nt)",
                threshold=int(long_threshold),
                size_scale=size_scale,
                rasterized=rasterized,
                show_legend=show_legend,
                title_fontsize=title_fontsize,
                legend_fontsize=max(4.5, legend_fontsize - 0.5),
            )
            continue
        if panel == "long":
            _style_panel(
                ax,
                title=f"(3) Long RNAs (>{long_threshold} nt)",
                xlim=(-1, 1),
                ylim=(-1, 1),
                title_fontsize=title_fontsize,
            )
            ax.text(
                0.5,
                0.5,
                "missing long t-SNE\n(run python run_rna_type.py plot)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="#666666",
            )
            continue
        if (
            panel == "rfam"
            and rfam_coords is not None
            and rfam_labels is not None
            and rfam_order is not None
        ):
            draw_rfam_on_ax(
                ax,
                rfam_coords,
                rfam_labels,
                rfam_order,
                title=PANEL_TITLES["rfam"],
                size_scale=size_scale,
                rasterized=rasterized,
                show_legend=show_legend,
                title_fontsize=title_fontsize,
                legend_fontsize=max(4.0, legend_fontsize - 1.0),
            )
            continue
        if panel == "rfam":
            _style_panel(
                ax,
                title=PANEL_TITLES["rfam"],
                xlim=(-1, 1),
                ylim=(-1, 1),
                title_fontsize=title_fontsize,
            )
            ax.text(
                0.5,
                0.5,
                "missing Rfam t-SNE\n(run python run_rna_type.py plot)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="#666666",
            )
            continue
        draw_functional_panel_on_ax(
            ax,
            panel,
            coords,
            labels,
            xlim=xlim,
            ylim=ylim,
            size_scale=size_scale,
            rasterized=rasterized,
            show_legend=show_legend,
            title_fontsize=title_fontsize,
            legend_fontsize=legend_fontsize,
        )


def plot_panels(
    coords: np.ndarray,
    labels: list[str],
    *,
    title: str,
    out_path: Path,
    formats: tuple[str, ...] = EXPORT_FORMATS,
    long_coords: np.ndarray | None = None,
    long_labels: list[str] | None = None,
    long_order: list[str] | None = None,
    long_threshold: int = 1024,
    rfam_coords: np.ndarray | None = None,
    rfam_labels: list[str] | None = None,
    rfam_order: list[str] | None = None,
) -> list[Path]:
    """2x2 panels; (3) long-seq and (4) Rfam when projections are provided."""
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.8), layout="constrained")
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13)
    draw_functional_quad_on_axes(
        axes,
        coords,
        labels,
        size_scale=1.0,
        rasterized=True,
        show_legend=True,
        title_fontsize=11,
        legend_fontsize=6.5,
        long_coords=long_coords,
        long_labels=long_labels,
        long_order=long_order,
        long_threshold=long_threshold,
        rfam_coords=rfam_coords,
        rfam_labels=rfam_labels,
        rfam_order=rfam_order,
    )
    axes[0, 0].set_title("(1) Functional classes", fontsize=11, pad=6)
    axes[0, 1].set_title("(2) Regulatory biotypes", fontsize=11, pad=6)
    axes[1, 0].set_title(
        f"(3) Long RNAs (>{long_threshold} nt)", fontsize=11, pad=6
    )
    axes[1, 1].set_title("(4) Rfam families", fontsize=11, pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = save_figure(
        fig,
        out_path,
        formats,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return written
