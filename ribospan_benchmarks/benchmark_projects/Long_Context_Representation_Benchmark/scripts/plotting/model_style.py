# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared model ordering, display names, and YaRN-aware color families."""

from __future__ import annotations

from typing import Iterable

YARN_SUFFIX = "-YaRN"

MODEL_ORDER = (
    "HydraRNA",
    "RiboGRAM",
    "RNAverse",
    "AIDO.RNA-CDS",
    "AIDO.RNA-CDS-YaRN",
    "RIBOSPAN-1K-15",
    "RIBOSPAN-1K-15-YaRN",
    "RIBOSPAN-1K-40",
    "RIBOSPAN-1K-40-YaRN",
    "RIBOSPAN-10K-15",
    "RIBOSPAN-10K-40",
)

DISPLAY_NAMES = {
    "AIDO.RNA-1.6B-CDS": "AIDO.RNA-CDS",
    "AIDO.RNA-1.6B-CDS-YaRN": "AIDO.RNA-CDS-YaRN",
    "RIBOSCOPE-1.6B-run4": "RIBOSPAN-1K-15",
    "RIBOSCOPE-1.6B-run4-YaRN": "RIBOSPAN-1K-15-YaRN",
    "RIBOSCOPE-1.6B-run4-2": "RIBOSPAN-1K-40",
    "RIBOSCOPE-1.6B-run4-2-YaRN": "RIBOSPAN-1K-40-YaRN",
    "RIBOSCOPE-1.6B-run5": "RIBOSPAN-10K-15",
    "RIBOSCOPE-1.6B-run5-2": "RIBOSPAN-10K-40",
}

# Bar colors of the long-context figure in the technical report. RiboGRAM and
# RNAverse are not in that figure; theirs are chosen from the same muted range.
BASE_MODEL_COLORS = {
    "HydraRNA": "#A58FA3",
    "AIDO.RNA-CDS": "#77736D",
    "RiboGRAM": "#6F9EA6",
    "RNAverse": "#A3A572",
    "RIBOSPAN-1K-15": "#596C9B",
    "RIBOSPAN-1K-40": "#557F73",
    "RIBOSPAN-10K-15": "#C2A454",
    "RIBOSPAN-10K-40": "#B87558",
    "AIDO.RNA-1.6B-CDS": "#77736D",
    "RIBOSCOPE-1.6B-run4": "#596C9B",
    "RIBOSCOPE-1.6B-run4-2": "#557F73",
    "RIBOSCOPE-1.6B-run5": "#C2A454",
    "RIBOSCOPE-1.6B-run5-2": "#B87558",
}

# Color of a model drawn next to its YaRN variant, which keeps the base color.
PAIRED_MODEL_COLORS = {
    "AIDO.RNA-CDS": "#C8C6C1",
    "RIBOSPAN-1K-15": "#A9B7D3",
    "RIBOSPAN-1K-40": "#A8C2B8",
    "AIDO.RNA-1.6B-CDS": "#C8C6C1",
    "RIBOSCOPE-1.6B-run4": "#A9B7D3",
    "RIBOSCOPE-1.6B-run4-2": "#A8C2B8",
}

LIGHTEN_AMOUNT = 0.5


def is_yarn_model(model: object) -> bool:
    return str(model).endswith(YARN_SUFFIX)


def base_model_name(model: object) -> str:
    name = str(model)
    if name.endswith(YARN_SUFFIX):
        return name[: -len(YARN_SUFFIX)]
    return name


def display_model_name(model: object) -> str:
    name = str(model)
    return DISPLAY_NAMES.get(name, name)


def lighten_hex(color: str, amount: float = LIGHTEN_AMOUNT) -> str:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    red = round(red + (255 - red) * amount)
    green = round(green + (255 - green) * amount)
    blue = round(blue + (255 - blue) * amount)
    return f"#{red:02X}{green:02X}{blue:02X}"


def ordered_models(values: Iterable[object]) -> list[str]:
    present = {str(value) for value in values}
    preferred = [model for model in MODEL_ORDER if model in present]
    remainder = sorted(present - set(preferred))
    return preferred + remainder


def model_color(model: object, *, present: set[str] | None = None) -> str | None:
    name = str(model)
    base = base_model_name(name)
    base_color = BASE_MODEL_COLORS.get(base)
    if base_color is None:
        return None
    if is_yarn_model(name):
        return base_color
    yarn_sibling = f"{base}{YARN_SUFFIX}"
    if present is not None and yarn_sibling in present:
        return PAIRED_MODEL_COLORS.get(base, lighten_hex(base_color))
    return base_color


def model_colors(values: Iterable[object]) -> dict[str, str]:
    models = ordered_models(values)
    present = set(models)
    colors: dict[str, str] = {}
    for index, model in enumerate(models):
        color = model_color(model, present=present)
        colors[model] = color if color is not None else f"C{index}"
    return colors


def display_model_colors(values: Iterable[object]) -> dict[str, str]:
    return {
        display_model_name(model): color
        for model, color in model_colors(values).items()
    }

