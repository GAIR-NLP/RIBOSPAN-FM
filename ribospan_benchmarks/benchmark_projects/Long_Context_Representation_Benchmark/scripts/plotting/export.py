# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Vector exports that keep text editable in Adobe Illustrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

VECTOR_EXTENSIONS = ("pdf", "svg")
CLEANUP_EXTENSIONS = ("png", "pdf", "svg")


def configure_vector_export() -> None:
    """Keep text as text instead of Type 3 outlines.

    Matplotlib's default ``pdf.fonttype=3`` writes Type 3 fonts that
    Illustrator cannot edit. Type 42 embeds TrueType; SVG with
    ``fonttype='none'`` stores real ``<text>`` elements.
    """

    import matplotlib as mpl  # type: ignore

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False


def save_figure(figure: Any, output_dir: Path, stem: str) -> list[str]:
    configure_vector_export()
    output_dir.mkdir(parents=True, exist_ok=True)
    leftover_png = output_dir / f"{stem}.png"
    if leftover_png.exists():
        leftover_png.unlink()
    paths: list[str] = []
    for extension in VECTOR_EXTENSIONS:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(
            path,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        paths.append(str(path))
    return paths


def save_figure_by_format(
    figure: Any, output_dir: Path, stem: str
) -> dict[str, str]:
    return {
        Path(path).suffix.lstrip("."): path
        for path in save_figure(figure, output_dir, stem)
    }
