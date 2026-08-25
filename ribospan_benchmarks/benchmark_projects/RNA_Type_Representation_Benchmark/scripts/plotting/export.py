# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Vector SVG/PDF export, plus a 1200 ppi hybrid SVG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

RASTER_DPI = 1200
EXPORT_FORMATS = ("svg", "pdf")
HYBRID_SUFFIX = "_raster"


def configure_vector_export() -> None:
    """Keep text as text instead of outlined paths / Type 3 fonts."""

    import matplotlib as mpl  # type: ignore

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False


def set_point_rasterized(figure: Any, rasterized: bool) -> None:
    """Toggle rasterization on scatter/collections, not on text or frames."""

    from matplotlib.collections import Collection  # type: ignore

    for artist in figure.findobj(Collection):
        artist.set_rasterized(bool(rasterized))


def save_figure(
    figure: Any,
    stem: Path | str,
    formats: Iterable[str] = EXPORT_FORMATS,
    *,
    write_hybrid_svg: bool = True,
    **savefig_kwargs: Any,
) -> list[Path]:
    configure_vector_export()
    stem_path = Path(stem).with_suffix("")
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    for leftover in (
        stem_path.with_suffix(".png"),
        Path(f"{stem_path}{HYBRID_SUFFIX}.png"),
    ):
        if leftover.exists():
            leftover.unlink()

    written: list[Path] = []
    vector_kwargs = dict(savefig_kwargs)
    vector_kwargs.pop("dpi", None)
    set_point_rasterized(figure, False)
    for fmt in formats:
        ext = str(fmt).lstrip(".").lower()
        if ext == "png":
            continue
        path = Path(f"{stem_path}.{ext}")
        figure.savefig(path, **vector_kwargs)
        written.append(path)

    if write_hybrid_svg:
        set_point_rasterized(figure, True)
        hybrid = Path(f"{stem_path}{HYBRID_SUFFIX}.svg")
        hybrid_kwargs = dict(savefig_kwargs)
        hybrid_kwargs["dpi"] = RASTER_DPI
        figure.savefig(hybrid, **hybrid_kwargs)
        written.append(hybrid)
    return written
