#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Multi-model composite: atlas + 2x2 panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze import (  # noqa: E402
    PLOT_EXCLUDE_TYPES,
    RNA_TYPE_COLORS,
    color_for_rna_type,
    load_ids,
    ordered_rna_types,
)
from scripts.config import display_model_name, load_experiment_config  # noqa: E402
from scripts.plotting.export import EXPORT_FORMATS, save_figure  # noqa: E402
from scripts.plotting.long import load_long_projection  # noqa: E402
from scripts.plotting.panels import (  # noqa: E402
    FUNCTIONAL_CLASS_COLORS,
    LONG_PANEL_COLORS,
    REGULATORY_PANEL_COLORS,
    draw_full_atlas_on_ax,
    draw_functional_quad_on_axes,
    rfam_color_map,
)
from scripts.plotting.rfam import load_rfam_projection  # noqa: E402


def _resolve_layer(emb_dir: Path, analysis_dir: Path, layer: int | None) -> int:
    """Prefer explicit layer; else meta.json layers[-1]; else existing tsne coords."""
    if layer is not None:
        return int(layer)
    meta_path = emb_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        layers = meta.get("layers") or []
        if layers:
            return int(layers[-1])
    matches = sorted(analysis_dir.glob("tsne_mean_layer*_coords.npy"))
    if matches:
        stem = matches[-1].name
        mid = stem.removeprefix("tsne_mean_layer").removesuffix("_coords.npy")
        return int(mid)
    raise FileNotFoundError(f"cannot resolve layer for {emb_dir} / {analysis_dir}")


def _short(name: str) -> str:
    return display_model_name(name)


def _load_tsne(analysis_dir: Path, emb_dir: Path, layer: int):
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    ids = load_ids(emb_dir / meta["ids_file"])
    coords = np.load(analysis_dir / f"tsne_mean_layer{layer}_coords.npy")
    idx = np.load(analysis_dir / f"tsne_mean_layer{layer}_index.npy")
    labels = [str(ids["rna_type"].iloc[int(i)]) for i in idx]
    return coords, labels


def _swatch_label(name: str, color: str, *, fontsize: float, markersize: float) -> HPacker:
    draw = DrawingArea(markersize * 1.5, markersize * 1.5, 0, 0)
    draw.add_artist(
        Circle(
            (markersize * 0.75, markersize * 0.75),
            markersize * 0.42,
            fc=color,
            ec="none",
        )
    )
    text = TextArea(
        name,
        textprops={"fontsize": fontsize, "va": "center", "fontfamily": "DejaVu Sans"},
    )
    return HPacker(children=[draw, text], align="center", pad=0, sep=2)


def _vsection(
    title: str,
    items: list[tuple[str, str]],
    *,
    title_fs: float,
    item_fs: float,
    markersize: float,
    item_sep: float = 1.8,
    ncol: int = 1,
) -> VPacker:
    title_box = TextArea(
        title,
        textprops={
            "fontsize": title_fs,
            "fontweight": "bold",
            "ha": "left",
            "fontfamily": "DejaVu Sans",
        },
    )
    swatches = [
        _swatch_label(name, color, fontsize=item_fs, markersize=markersize)
        for name, color in items
    ]
    ncol = max(1, int(ncol))
    if ncol == 1:
        body = VPacker(children=swatches, align="left", pad=0, sep=item_sep)
    else:
        cols: list[list] = [[] for _ in range(ncol)]
        for j, sw in enumerate(swatches):
            cols[j % ncol].append(sw)
        body = HPacker(
            children=[
                VPacker(children=col, align="left", pad=0, sep=item_sep)
                for col in cols
                if col
            ],
            align="top",
            pad=0,
            sep=5,
        )
    return VPacker(children=[title_box, body], align="left", pad=0, sep=item_sep)


def _atlas_items(atlas_types: list[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    fallback_i = 0
    for name in atlas_types:
        if name in PLOT_EXCLUDE_TYPES:
            continue
        color = color_for_rna_type(name, fallback_i)
        if name not in RNA_TYPE_COLORS:
            fallback_i += 1
        items.append((name, color))
    return items


def _atlas_legend_box(atlas_types: list[str]) -> VPacker:
    """Compact multi-column type legend for the top-row atlas, titled RNA Types."""
    items = _atlas_items(atlas_types)
    cols_n = 8
    cols: list[list] = [[] for _ in range(cols_n)]
    for j, (name, color) in enumerate(items):
        cols[j % cols_n].append(
            _swatch_label(name, color, fontsize=5.6, markersize=4.2)
        )
    body = HPacker(
        children=[
            VPacker(children=col, align="left", pad=0, sep=1.6)
            for col in cols
            if col
        ],
        align="top",
        pad=0,
        sep=6,
    )
    title = TextArea(
        "RNA Types",
        textprops={
            "fontsize": 7.2,
            "fontweight": "bold",
            "ha": "left",
            "fontfamily": "DejaVu Sans",
        },
    )
    return VPacker(children=[title, body], align="left", pad=0, sep=1.8)


def _panel_legend_box(
    long_order: list[str],
    rfam_order: list[str],
    *,
    long_threshold: int = 1024,
) -> HPacker:
    """Centered (1)–(4) panel legend block."""
    functional_items = [
        (k, FUNCTIONAL_CLASS_COLORS[k]) for k in ("housekeeping", "regulatory", "coding")
    ]
    long_items = [
        (lab, LONG_PANEL_COLORS.get(lab, color_for_rna_type(lab, i)))
        for i, lab in enumerate(long_order)
    ]
    reg = [
        (k, REGULATORY_PANEL_COLORS[k])
        for k in (
            "lncRNA",
            "snRNA",
            "snoRNA",
            "miRNA",
            "pre_miRNA",
            "piRNA",
            "siRNA",
        )
    ]
    color_map = rfam_color_map(rfam_order)
    rfam_items = [(lab, color_map[lab]) for lab in rfam_order]

    title_fs = 7.2
    item_fs = 6.0
    rem_fs = 5.4
    ms = 4.6
    item_sep = 1.6

    sec1 = _vsection(
        "(1) Functional classes",
        functional_items,
        title_fs=title_fs,
        item_fs=item_fs,
        markersize=ms,
        item_sep=item_sep,
    )
    sec2 = _vsection(
        "(2) Regulatory biotypes",
        reg,
        title_fs=title_fs,
        item_fs=item_fs,
        markersize=ms,
        item_sep=item_sep,
        ncol=2,
    )
    sec3 = _vsection(
        f"(3) Long RNAs (>{long_threshold} nt)",
        long_items,
        title_fs=title_fs,
        item_fs=item_fs,
        markersize=ms,
        item_sep=item_sep,
        ncol=2,
    )

    rem_cols_n = 4
    rem_cols: list[list] = [[] for _ in range(rem_cols_n)]
    for j, (name, color) in enumerate(rfam_items):
        rem_cols[j % rem_cols_n].append(
            _swatch_label(name, color, fontsize=rem_fs, markersize=4.0)
        )
    rem_body = HPacker(
        children=[
            VPacker(children=col, align="left", pad=0, sep=item_sep)
            for col in rem_cols
            if col
        ],
        align="top",
        pad=0,
        sep=5,
    )
    rem_title = TextArea(
        "(4) Rfam families",
        textprops={
            "fontsize": title_fs,
            "fontweight": "bold",
            "ha": "left",
            "fontfamily": "DejaVu Sans",
        },
    )
    sec4 = VPacker(children=[rem_title, rem_body], align="left", pad=0, sep=item_sep)

    return HPacker(
        children=[sec1, sec2, sec3, sec4],
        align="top",
        pad=0,
        sep=11,
    )


def _offsetbox_size_inches(box: Any) -> tuple[float, float]:
    """Measure an OffsetBox in inches using a throwaway Agg renderer."""

    from matplotlib.offsetbox import AnnotationBbox

    fig = plt.figure(figsize=(40, 10), dpi=100)
    fig.add_artist(
        AnnotationBbox(
            box,
            (0.5, 0.5),
            xycoords="figure fraction",
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0.0,
        )
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = box.get_window_extent(renderer)
    dpi = float(fig.dpi)
    plt.close(fig)
    return float(bbox.width) / dpi, float(bbox.height) / dpi


def plot_composite(
    *,
    output_root: Path,
    model_names: list[str],
    layer: int | None,
    out_path: Path,
    formats: tuple[str, ...],
    map_df: pd.DataFrame | None = None,
    long_threshold: int = 1024,
) -> list[Path]:
    atlas_present: set[str] = set()
    loaded: list[tuple] = []
    rfam_order_global: list[str] | None = None
    long_order_global: list[str] | None = None
    thr_used = int(long_threshold)
    for name in model_names:
        analysis_dir = output_root / "analysis" / name
        emb_dir = output_root / "embeddings" / name
        try:
            resolved = _resolve_layer(emb_dir, analysis_dir, layer)
        except FileNotFoundError:
            print(f"skip missing t-SNE coords for {name}", flush=True)
            continue
        if not (analysis_dir / f"tsne_mean_layer{resolved}_coords.npy").exists():
            print(f"skip missing t-SNE coords for {name}", flush=True)
            continue
        coords, labels = _load_tsne(analysis_dir, emb_dir, resolved)
        atlas_present.update(lab for lab in labels if lab not in PLOT_EXCLUDE_TYPES)

        long_loaded = load_long_projection(analysis_dir, resolved)
        if long_loaded is None:
            print(f"skip missing long t-SNE for {name}", flush=True)
            continue
        long_coords, long_labels, long_order, thr = long_loaded
        thr_used = int(thr)
        if long_order_global is None:
            long_order_global = list(long_order)

        rfam_loaded = load_rfam_projection(analysis_dir, resolved, map_df=map_df)
        if rfam_loaded is None:
            print(f"skip missing Rfam t-SNE for {name}", flush=True)
            continue
        rfam_coords, rfam_labels, rfam_order = rfam_loaded
        if rfam_order_global is None:
            rfam_order_global = list(rfam_order)
        loaded.append(
            (
                name,
                coords,
                labels,
                long_coords,
                long_labels,
                long_order,
                thr,
                rfam_coords,
                rfam_labels,
                rfam_order,
            )
        )

    if not loaded:
        print("skip composite: no models with t-SNE", flush=True)
        return []

    n = len(loaded)
    col_w = 3.6 if n <= 6 else max(2.8, 22.0 / n)
    fig_w_cols = col_w * n + 0.4
    fig_h = col_w * 2 + 2.70

    atlas_types = ordered_rna_types(sorted(atlas_present))
    atlas_w, _ = _offsetbox_size_inches(_atlas_legend_box(atlas_types))
    panel_w, _ = _offsetbox_size_inches(
        _panel_legend_box(
            long_order_global or [],
            rfam_order_global or [],
            long_threshold=thr_used,
        )
    )
    legend_w = max(atlas_w, panel_w) + 0.8
    fig_w = max(fig_w_cols, legend_w)
    extra = fig_w - fig_w_cols
    left = (0.015 * fig_w_cols + extra / 2.0) / fig_w
    right = 1.0 - (0.005 * fig_w_cols + extra / 2.0) / fig_w
    print(
        f"composite canvas: models={n} fig_w={fig_w:.2f}in "
        f"(columns={fig_w_cols:.2f}, atlas_legend={atlas_w:.2f}, "
        f"panel_legend={panel_w:.2f})",
        flush=True,
    )

    fig = plt.figure(figsize=(fig_w, fig_h), layout=None)
    fig.patch.set_facecolor("white")

    outer = GridSpec(
        3,
        n,
        figure=fig,
        height_ratios=[1.0, 0.24, 1.0],
        width_ratios=[1.0] * n,
        hspace=0.05,
        wspace=0.14,
        left=left,
        right=right,
        top=0.978,
        bottom=0.145,
    )

    atlas_axes: list = []
    facet_top_axes: list = []
    facet_bottom_axes: list = []
    for col, (
        name,
        coords,
        labels,
        long_coords,
        long_labels,
        long_order,
        thr,
        rfam_coords,
        rfam_labels,
        rfam_order,
    ) in enumerate(loaded):
        ax_full = fig.add_subplot(outer[0, col])
        draw_full_atlas_on_ax(
            ax_full,
            coords,
            labels,
            title=_short(name),
            size_scale=None,
            rasterized=True,
            show_legend=False,
            title_fontsize=12,
        )
        atlas_axes.append(ax_full)

        inner = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[2, col],
            wspace=0.08,
            hspace=0.16,
        )
        axes = np.empty((2, 2), dtype=object)
        for r in range(2):
            for c in range(2):
                axes[r, c] = fig.add_subplot(inner[r, c])
        draw_functional_quad_on_axes(
            axes,
            coords,
            labels,
            size_scale=None,
            rasterized=True,
            show_legend=False,
            title_fontsize=7.5,
            legend_fontsize=4.5,
            long_coords=long_coords,
            long_labels=long_labels,
            long_order=long_order,
            long_threshold=int(thr),
            rfam_coords=rfam_coords,
            rfam_labels=rfam_labels,
            rfam_order=rfam_order,
        )
        facet_top_axes.extend([axes[0, 0], axes[0, 1]])
        facet_bottom_axes.extend([axes[1, 0], axes[1, 1]])

    # Keep mid-row slot for atlas↔panel spacing (no content).
    ax_mid = fig.add_subplot(outer[1, :])
    ax_mid.set_axis_off()

    # Place atlas legend in the mid gap, slightly above center (closer to atlases).
    fig.canvas.draw()
    atlas_y0 = min(ax.get_position().y0 for ax in atlas_axes)
    facet_y1 = max(ax.get_position().y1 for ax in facet_top_axes)
    gap = atlas_y0 - facet_y1
    y_atlas_leg = facet_y1 + 0.62 * gap

    fig.add_artist(
        AnnotationBbox(
            _atlas_legend_box(atlas_types),
            (0.5, y_atlas_leg),
            xycoords="figure fraction",
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0.0,
        )
    )

    facet_y0 = min(ax.get_position().y0 for ax in facet_bottom_axes)
    y_panel_leg = 0.5 * facet_y0
    fig.add_artist(
        AnnotationBbox(
            _panel_legend_box(
                long_order_global or [],
                rfam_order_global or [],
                long_threshold=thr_used,
            ),
            (0.5, y_panel_leg),
            xycoords="figure fraction",
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0.0,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = save_figure(
        fig,
        out_path,
        formats,
        facecolor=fig.get_facecolor(),
    )
    for path in written:
        print(f"wrote {path}", flush=True)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="t-SNE layer index. Default: each model's meta.json layers[-1].",
    )
    parser.add_argument("--formats", nargs="+", default=list(EXPORT_FORMATS))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output stem path (.svg/.pdf added). Default: analysis/composite_tsne_mean",
    )
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    output_root = (project_root / cfg["output_root"]).resolve()
    model_names = args.models or list(cfg["models"]["names"])
    canonical = list(cfg["models"]["names"])
    default_out = output_root / "analysis" / "composite_tsne_mean.pdf"
    if args.out is None:
        if args.models is not None and list(model_names) != list(canonical):
            raise SystemExit(
                "refusing to overwrite analysis/composite_tsne_mean with a "
                "non-canonical model list; pass --out"
            )
        out = default_out
    else:
        out = args.out
    long_threshold = int(cfg.get("analysis", {}).get("long_length_threshold", 1024))

    map_path = output_root / "rfam_mapping" / "seq_to_rfam.csv"
    map_df = pd.read_csv(map_path) if map_path.exists() else None

    plot_composite(
        output_root=output_root,
        model_names=model_names,
        layer=args.layer,
        out_path=out,
        formats=tuple(args.formats),
        map_df=map_df,
        long_threshold=long_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
