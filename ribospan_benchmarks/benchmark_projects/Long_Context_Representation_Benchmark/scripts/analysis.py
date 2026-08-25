#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Analyze long-context representation endpoints and attention diagnostics."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import load_experiment_config, public_path
from .plotting.attention import plot_attention_diagnostics
from .plotting.cosine import plot_cosine_diagnostics
from .plotting.model_style import display_model_name, ordered_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_BH_ENDPOINTS = (
    "final_context_gain",
    "final_cross_region_same_base_cosine",
    "final_relative_distal_075",
)
ATTENTION_BH_ENDPOINTS = ("late_layer_attention_convergence",)
BH_ENDPOINTS = set(PRIMARY_BH_ENDPOINTS) | set(ATTENTION_BH_ENDPOINTS)
SENSITIVITY_PAIRED_ENDPOINTS = (
    "final_relative_distal_025",
    "final_relative_distal_050",
    "final_far_field_diffusion",
    "final_relative_distal_diffusion",
    "late_layer_attention_convergence",
)
CONTEXT_CONTROL_METRICS = (
    "context_separation",
    "cross_context_same_base_cos",
    "within_motif_same_base_cos",
    "within_background_same_base_cos",
    "same_context_baseline_cos",
)
PAPER_METRIC_COLUMNS = (
    ("delta_cs", "delta_context_separation_structured_vs_native"),
    ("c_cross", "cross_context_same_base_cos_structured"),
    ("d_distal", "relative_distal_075"),
)
PAIRED_ENDPOINT_LABELS = {
    "final_context_gain": "ΔCS",
    "final_cross_region_same_base_cosine": "C_cross",
    "final_relative_distal_075": "D_distal",
    "final_relative_distal_diffusion": "D_distal (config alias)",
    "final_relative_distal_025": "relative-distal r≥0.25",
    "final_relative_distal_050": "relative-distal r≥0.50",
    "final_far_field_diffusion": "far-field ≥1024 nt",
    "late_layer_attention_convergence": "late-layer attention distance",
}
LENGTH_PRIMARY_HEADERS = {
    "delta_cs": "ΔCS",
    "delta_cs_ci_low": "ΔCS_ci_low",
    "delta_cs_ci_high": "ΔCS_ci_high",
    "c_cross": "C_cross",
    "c_cross_ci_low": "C_cross_ci_low",
    "c_cross_ci_high": "C_cross_ci_high",
    "d_distal": "D_distal",
    "d_distal_ci_low": "D_distal_ci_low",
    "d_distal_ci_high": "D_distal_ci_high",
}
DIAGNOSTIC_HEADERS = {
    "context_separation": "CS_structured",
    "context_separation_ci_low": "CS_structured_ci_low",
    "context_separation_ci_high": "CS_structured_ci_high",
    "far_field_diffusion": "far_field_ge_1024",
    "far_field_diffusion_ci_low": "far_field_ci_low",
    "far_field_diffusion_ci_high": "far_field_ci_high",
    "relative_distal_025_mean": "relative_distal_r025",
    "relative_distal_050_mean": "relative_distal_r050",
    "relative_distal_075_mean": "D_distal_length_pooled",
    "layer_max_contextualization": "layer_max_CS_structured",
    "layer_max_leakage": "layer_max_far_field",
}
CONVERGENCE_HEADERS = {
    "delta_context_separation_structured_vs_native_late_minus_middle": (
        "ΔCS_late_minus_middle"
    ),
    "context_separation_structured_late_minus_middle": (
        "CS_structured_late_minus_middle"
    ),
    "cross_context_same_base_cos_structured_late_minus_middle": (
        "C_cross_late_minus_middle"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--cosine-dir", default=None)
    parser.add_argument("--attention-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260709)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _read_csv(path: Path, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty and required:
        raise ValueError(f"Required result table is empty: {path}")
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _use_length_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Analyze full transcripts by target length bin while retaining actual length."""

    if frame.empty or "length_group" not in frame:
        return frame
    result = frame.copy()
    result["actual_sequence_length"] = pd.to_numeric(
        result.get("actual_length", result["length"]), errors="coerce"
    )
    result["length"] = pd.to_numeric(
        result["length_group"], errors="coerce"
    )
    return result


def _bootstrap_mean(
    values: Sequence[float],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return (math.nan, math.nan, math.nan, 0)
    estimate = float(array.mean())
    if len(array) == 1 or replicates < 2:
        return (estimate, math.nan, math.nan, int(len(array)))
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    draws = array[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high), int(len(array))


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    replicates: int,
    rng: np.random.Generator,
    *,
    cluster_column: str = "seed",
) -> tuple[float, float, float, int]:
    """Bootstrap equal-weight cluster means instead of correlated job rows."""

    if frame.empty or value_column not in frame:
        return (math.nan, math.nan, math.nan, 0)
    clean = frame.dropna(subset=[value_column]).copy()
    if clean.empty:
        return (math.nan, math.nan, math.nan, 0)
    if cluster_column not in clean:
        return _bootstrap_mean(
            clean[value_column].to_numpy(), replicates, rng
        )
    cluster_means = (
        clean.groupby(cluster_column, dropna=False)[value_column]
        .mean()
        .to_numpy()
    )
    return _bootstrap_mean(cluster_means, replicates, rng)


def _paired_bootstrap(
    differences: Sequence[float],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int]:
    array = np.asarray(differences, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return (math.nan, math.nan, math.nan, math.nan, 0)
    estimate = float(array.mean())
    if len(array) == 1 or replicates < 2:
        return (estimate, math.nan, math.nan, math.nan, int(len(array)))
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    draws = array[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    p_value = min(
        1.0,
        2.0
        * min(
            (float(np.count_nonzero(draws <= 0)) + 1.0) / (replicates + 1.0),
            (float(np.count_nonzero(draws >= 0)) + 1.0) / (replicates + 1.0),
        ),
    )
    return estimate, float(low), float(high), p_value, int(len(array))


def _order_models_in_frame(
    frame: pd.DataFrame,
    column: str = "model",
    extra_sort: Sequence[str] = (),
) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return frame
    rank = {
        name: index for index, name in enumerate(ordered_models(frame[column]))
    }
    result = frame.copy()
    result["_model_order"] = result[column].map(rank)
    sort_columns = [*(extra_sort), "_model_order"]
    result = result.sort_values(sort_columns, kind="mergesort").drop(
        columns="_model_order"
    )
    return result.reset_index(drop=True)


def _pair_level_paper_metrics(
    context_controls: pd.DataFrame,
    diffusion_final: pd.DataFrame,
) -> pd.DataFrame:
    """One final-layer ΔCS, C_cross, and D_distal record per transcript pair."""

    required_context = {
        "model",
        "length",
        "pair_group_id",
        "layer",
        "delta_context_separation_structured_vs_native",
        "cross_context_same_base_cos_structured",
    }
    required_diffusion = {
        "model",
        "length",
        "pair_group_id",
        "relative_distal_075",
    }
    if (
        context_controls.empty
        or diffusion_final.empty
        or required_context - set(context_controls)
        or required_diffusion - set(diffusion_final)
    ):
        return pd.DataFrame()
    context = context_controls.copy()
    context["length"] = pd.to_numeric(context["length"], errors="coerce")
    diffusion = diffusion_final.copy()
    diffusion["length"] = pd.to_numeric(diffusion["length"], errors="coerce")
    final_layer = context.groupby(
        ["model", "length", "pair_group_id"]
    )["layer"].transform("max")
    context = context[context["layer"] == final_layer]
    context_pairs = (
        context.groupby(
            ["model", "length", "pair_group_id"],
            as_index=False,
        )[
            [
                "delta_context_separation_structured_vs_native",
                "cross_context_same_base_cos_structured",
            ]
        ]
        .mean()
    )
    diffusion_pairs = (
        diffusion.groupby(
            ["model", "length", "pair_group_id"],
            as_index=False,
        )["relative_distal_075"]
        .mean()
    )
    return context_pairs.merge(
        diffusion_pairs,
        on=["model", "length", "pair_group_id"],
        how="inner",
    )


def _length_primary_scores(
    pair_metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap ΔCS, C_cross, and D_distal at the transcript-pair grain."""

    if pair_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    models = ordered_models(pair_metrics["model"])
    lengths = sorted(
        int(value) for value in pair_metrics["length"].dropna().unique()
    )
    for length_index, length in enumerate(lengths):
        for model_index, model in enumerate(models):
            subset = pair_metrics[
                (pair_metrics["model"] == model)
                & (pair_metrics["length"] == length)
            ]
            rng = np.random.default_rng(
                seed + 7919 * length_index + 1009 * model_index
            )
            row: dict[str, Any] = {
                "model": model,
                "length": length,
                "n_pairs": (
                    int(subset["pair_group_id"].nunique())
                    if not subset.empty
                    else 0
                ),
            }
            for out_name, column in PAPER_METRIC_COLUMNS:
                values = (
                    subset[column].to_numpy(dtype=float)
                    if column in subset and not subset.empty
                    else np.asarray([], dtype=float)
                )
                stats = _bootstrap_mean(values, replicates, rng)
                row[out_name] = stats[0]
                row[f"{out_name}_ci_low"] = stats[1]
                row[f"{out_name}_ci_high"] = stats[2]
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_tests_for_report(
    frame: pd.DataFrame,
    endpoints: Sequence[str],
) -> pd.DataFrame:
    if frame.empty or "endpoint" not in frame:
        return pd.DataFrame()
    selected = frame[frame["endpoint"].isin(endpoints)].copy()
    if selected.empty:
        return selected
    rank = {name: index for index, name in enumerate(endpoints)}
    selected["_endpoint_order"] = selected["endpoint"].map(rank)
    selected = selected.sort_values(
        ["_endpoint_order", "length", "model_a", "model_b"],
        kind="mergesort",
    ).drop(columns="_endpoint_order")
    selected["endpoint"] = selected["endpoint"].map(
        lambda value: PAIRED_ENDPOINT_LABELS.get(str(value), str(value))
    )
    return selected.reset_index(drop=True)


def _bh_fdr(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    adjusted = np.full(len(array), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(array))
    if not len(finite_indices):
        return adjusted.tolist()
    order = finite_indices[np.argsort(array[finite_indices])]
    count = len(order)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, float(array[index]) * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _assign_grouped_q_values(result: pd.DataFrame) -> pd.DataFrame:
    """BH-adjust p-values within selected endpoints; leave others as NaN."""

    frame = result.copy()
    frame["q_value_bh"] = np.nan
    if frame.empty or "endpoint" not in frame or "p_value" not in frame:
        return frame
    for endpoint, group in frame.groupby("endpoint"):
        if endpoint not in BH_ENDPOINTS:
            continue
        frame.loc[group.index, "q_value_bh"] = _bh_fdr(group["p_value"].to_numpy())
    return frame


def _final_context(context: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "job_id",
        "control_type",
        "metric_variant",
        "layer",
        "base",
        "context_separation",
    }
    missing = required - set(context)
    if missing:
        raise ValueError(f"cosine_context.csv misses columns: {sorted(missing)}")
    context = _numeric(
        context,
        ("layer", "length", "seed", "strength_value", "context_separation"),
    )
    selected = context[
        (context["control_type"] == "structured")
        & (context["metric_variant"] == "raw")
    ].copy()
    keys = [
        column
        for column in (
            "model",
            "job_id",
            "pair_group_id",
            "seed",
            "length",
            "actual_sequence_length",
            "source_type",
            "background_type",
            "pattern_id",
            "pattern_family",
            "strength_mode",
            "strength_value",
            "motif_start",
            "motif_end",
            "layer",
        )
        if column in selected
    ]
    by_layer = (
        selected.groupby(keys, dropna=False, as_index=False)["context_separation"]
        .mean()
        .dropna(subset=["context_separation"])
    )
    final_layer = by_layer.groupby("model")["layer"].transform("max")
    return by_layer[by_layer["layer"] == final_layer].copy()


def _final_diffusion(diffusion: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "pair_group_id",
        "control_type",
        "metric_variant",
        "layer",
        "far_field",
        "relative_distal",
        "relative_distal_025",
        "relative_distal_050",
        "relative_distal_075",
        "leakage_auc",
    }
    missing = required - set(diffusion)
    if missing:
        raise ValueError(f"cosine_diffusion.csv misses columns: {sorted(missing)}")
    diffusion = _numeric(
        diffusion,
        (
            "layer",
            "length",
            "actual_sequence_length",
            "seed",
            "strength_value",
            "far_field",
            "relative_distal",
            "relative_distal_025",
            "relative_distal_050",
            "relative_distal_075",
            "leakage_auc",
            "local_peak",
        ),
    )
    diffusion["control_type"] = (
        diffusion["control_type"]
        .astype(str)
        .str.lower()
        .replace({"none": "native"})
    )
    selected = diffusion[
        (diffusion["control_type"] == "native")
        & (diffusion["metric_variant"] == "raw")
    ].copy()
    dedupe = [
        column
        for column in (
            "model",
            "pair_group_id",
            "control_job_id",
            "seed",
            "length",
            "actual_sequence_length",
            "source_type",
            "background_type",
            "pattern_id",
            "pattern_family",
            "strength_mode",
            "strength_value",
            "motif_start",
            "motif_end",
            "layer",
            "metric_variant",
        )
        if column in selected
    ]
    by_layer = selected.drop_duplicates(dedupe).copy()
    if by_layer.empty:
        return by_layer
    final_layer = by_layer.groupby("model")["layer"].transform("max")
    return by_layer[by_layer["layer"] == final_layer].copy()


def _context_control_comparison(context: pd.DataFrame) -> pd.DataFrame:
    """Pair structured geometry with the unmodified native control."""

    required = {
        "model",
        "pair_group_id",
        "control_type",
        "metric_variant",
        "layer",
        "base",
        "context_separation",
    }
    missing = required - set(context)
    if missing:
        raise ValueError(
            f"cosine_context.csv misses control-comparison columns: {sorted(missing)}"
        )
    metrics = [column for column in CONTEXT_CONTROL_METRICS if column in context]
    selected = context[
        context["metric_variant"].astype(str).str.lower() == "raw"
    ].copy()
    selected["control_type"] = (
        selected["control_type"].astype(str).str.lower()
    )
    selected["control_type"] = selected["control_type"].replace(
        {"none": "native"}
    )
    selected = _numeric(selected, ("layer", "length", *metrics))
    keys = [
        column
        for column in (
            "model",
            "pair_group_id",
            "seed",
            "length",
            "actual_sequence_length",
            "source_type",
            "background_type",
            "pattern_id",
            "pattern_family",
            "strength_mode",
            "strength_value",
            "motif_start",
            "motif_end",
            "layer",
            "base",
        )
        if column in selected
    ]
    grouped = (
        selected.groupby(
            [*keys, "control_type"], as_index=False, dropna=False
        )[metrics]
        .mean()
    )
    if grouped.empty:
        return pd.DataFrame()
    wide = grouped.set_index([*keys, "control_type"])[metrics].unstack(
        "control_type"
    )
    wide.columns = [
        f"{metric}_{control}" for metric, control in wide.columns
    ]
    wide = wide.reset_index()
    controls = (
        grouped.groupby(keys, as_index=False, dropna=False)["control_type"]
        .nunique()
        .rename(columns={"control_type": "n_controls_present"})
    )
    wide = wide.merge(controls, on=keys, how="left")

    for metric in metrics:
        structured = f"{metric}_structured"
        if structured not in wide:
            continue
        for control in ("native",):
            control_column = f"{metric}_{control}"
            if control_column not in wide:
                continue
            delta = f"delta_{metric}_structured_vs_{control}"
            wide[delta] = wide[structured] - wide[control_column]
    return wide.sort_values(
        [column for column in ("model", "length", "pair_group_id", "layer", "base") if column in wide]
    )


def _context_layer_dynamics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bases and calculate adjacent-layer representation changes."""

    if comparison.empty:
        return pd.DataFrame()
    metrics = [
        column
        for column in comparison
        if (
            column.startswith("delta_")
            or any(
                column == f"{metric}_{control}"
                for metric in CONTEXT_CONTROL_METRICS
                for control in ("structured", "native")
            )
        )
    ]
    keys = [
        column
        for column in (
            "model",
            "pair_group_id",
            "seed",
            "length",
            "actual_sequence_length",
            "source_type",
            "background_type",
            "pattern_id",
            "pattern_family",
            "strength_mode",
            "strength_value",
            "layer",
        )
        if column in comparison
    ]
    layer = (
        comparison.groupby(keys, as_index=False, dropna=False)
        .agg(
            **{column: (column, "mean") for column in metrics},
            n_bases_with_estimate=("base", "nunique"),
        )
        .sort_values(
            [
                column
                for column in ("model", "pair_group_id", "layer")
                if column in keys
            ]
        )
    )
    unit_keys = [column for column in keys if column != "layer"]
    for metric in metrics:
        layer[f"layer_delta_{metric}"] = layer.groupby(
            unit_keys, dropna=False
        )[metric].diff()
    return layer


def _late_layer_summary(
    dynamics: pd.DataFrame,
    *,
    layer_column: str = "layer",
) -> pd.DataFrame:
    """Compare middle-quarter integration with final-quarter convergence."""

    if dynamics.empty or layer_column not in dynamics:
        return pd.DataFrame()
    unit_keys = [
        column
        for column in (
            "model",
            "pair_group_id",
            "job_id",
            "seed",
            "length",
            "actual_sequence_length",
            "source_type",
            "background_type",
            "pattern_id",
            "pattern_family",
            "strength_mode",
            "strength_value",
        )
        if column in dynamics
    ]
    metric_columns = [
        column
        for column in dynamics
        if column
        not in {
            *unit_keys,
            layer_column,
            "attention_layer",
            "hidden_layer",
            "n_bases_with_estimate",
        }
        and not column.startswith("layer_delta_")
        and pd.api.types.is_numeric_dtype(dynamics[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in dynamics.groupby(unit_keys, dropna=False):
        clean = group.dropna(subset=[layer_column]).copy()
        if clean.empty:
            continue
        maximum = int(clean[layer_column].max())
        if maximum < 4:
            continue
        middle_start = max(1, int(math.ceil(maximum * 0.25)))
        middle_end = max(middle_start, int(math.floor(maximum * 0.50)))
        late_start = max(1, int(math.ceil(maximum * 0.75)))
        middle = clean[
            clean[layer_column].between(middle_start, middle_end)
        ]
        late = clean[clean[layer_column].between(late_start, maximum)]
        if middle.empty or late.empty:
            continue
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(unit_keys, key_values))
        row.update(
            {
                "middle_layer_start": middle_start,
                "middle_layer_end": middle_end,
                "late_layer_start": late_start,
                "late_layer_end": maximum,
            }
        )
        for metric in metric_columns:
            middle_mean = float(middle[metric].mean())
            late_mean = float(late[metric].mean())
            if math.isfinite(middle_mean) and math.isfinite(late_mean):
                row[f"{metric}_middle_mean"] = middle_mean
                row[f"{metric}_late_mean"] = late_mean
                row[f"{metric}_late_minus_middle"] = late_mean - middle_mean
        rows.append(row)
    return pd.DataFrame(rows)


def _attention_layer_dynamics(summary: pd.DataFrame) -> pd.DataFrame:
    """Create Attention_Heatmap-compatible per-job layer trajectories."""

    if summary.empty:
        return pd.DataFrame()
    required = {
        "model_name",
        "job_id",
        "pair_group_id",
        "layer",
        "head",
        "mean_distance",
        "normalized_mean_distance",
    }
    if required - set(summary):
        return pd.DataFrame()
    frame = summary.copy()
    if "control_type" in frame:
        structured = frame[
            frame["control_type"].astype(str).str.lower() == "structured"
        ]
        if not structured.empty:
            frame = structured.copy()
    bin_columns = [
        column
        for column in (
            "bin_513_1024",
            "bin_1025_2048",
            "bin_2049_4096",
            "bin_4097_plus",
        )
        if column in frame
    ]
    frame = _numeric(
        frame,
        (
            "layer",
            "length",
            "mean_distance",
            "normalized_mean_distance",
            *bin_columns,
        ),
    )
    if {"bin_1025_2048", "bin_2049_4096", "bin_4097_plus"}.issubset(frame):
        frame["mass_gt_1024"] = frame[
            ["bin_1025_2048", "bin_2049_4096", "bin_4097_plus"]
        ].sum(axis=1)
    if {"bin_2049_4096", "bin_4097_plus"}.issubset(frame):
        frame["mass_gt_2048"] = frame[
            ["bin_2049_4096", "bin_4097_plus"]
        ].sum(axis=1)
    metrics = [
        column
        for column in (
            "mean_distance",
            "normalized_mean_distance",
            "mass_gt_1024",
            "mass_gt_2048",
            "attention_entropy",
            "motif_attention_enrichment",
        )
        if column in frame
    ]
    keys = [
        column
        for column in (
            "model_name",
            "job_id",
            "pair_group_id",
            "seed",
            "length",
            "actual_sequence_length",
            "source_type",
            "background_type",
            "pattern_id",
            "pattern_family",
            "strength_mode",
            "strength_value",
            "layer",
        )
        if column in frame
    ]
    result = (
        frame.groupby(keys, as_index=False, dropna=False)[metrics]
        .mean()
        .rename(columns={"model_name": "model", "layer": "attention_layer"})
    )
    result["hidden_layer"] = result["attention_layer"] + 1
    unit_keys = [
        column
        for column in result
        if column not in {"attention_layer", "hidden_layer", *metrics}
    ]
    result = result.sort_values(
        ["model", "pair_group_id", "attention_layer"]
    )
    for metric in metrics:
        result[f"layer_delta_{metric}"] = result.groupby(
            unit_keys, dropna=False
        )[metric].diff()
    return result


def _attention_similarity_association(
    attention_dynamics: pd.DataFrame,
    context_dynamics: pd.DataFrame,
) -> pd.DataFrame:
    """Align attention block l with the hidden state after that block (l+1)."""

    if attention_dynamics.empty or context_dynamics.empty:
        return pd.DataFrame()
    right = context_dynamics.rename(columns={"layer": "hidden_layer"})
    merge_keys = [
        column
        for column in ("model", "pair_group_id", "length", "hidden_layer")
        if column in attention_dynamics and column in right
    ]
    if not {"model", "pair_group_id", "hidden_layer"}.issubset(merge_keys):
        return pd.DataFrame()
    context_metrics = [
        column
        for column in right
        if column.startswith("context_separation_")
        or column.startswith("cross_context_same_base_cos_")
        or column.startswith("delta_context_separation_")
        or column.startswith("layer_delta_context_separation_")
        or column.startswith("layer_delta_cross_context_same_base_cos_")
        or column.startswith("layer_delta_delta_context_separation_")
    ]
    context_view = right[[*merge_keys, *context_metrics]].drop_duplicates(
        merge_keys
    )
    return attention_dynamics.merge(
        context_view, on=merge_keys, how="inner", validate="many_to_one"
    )


def _association_correlations(association: pd.DataFrame) -> pd.DataFrame:
    if association.empty:
        return pd.DataFrame()
    pairs = (
        (
            "normalized_mean_distance",
            "cross_context_same_base_cos_structured",
            "layer_level",
        ),
        (
            "normalized_mean_distance",
            "context_separation_structured",
            "layer_level",
        ),
        (
            "normalized_mean_distance",
            "delta_context_separation_structured_vs_native",
            "layer_level",
        ),
        (
            "layer_delta_normalized_mean_distance",
            "layer_delta_cross_context_same_base_cos_structured",
            "adjacent_layer_change",
        ),
        (
            "layer_delta_normalized_mean_distance",
            "layer_delta_context_separation_structured",
            "adjacent_layer_change",
        ),
        (
            "layer_delta_normalized_mean_distance",
            "layer_delta_delta_context_separation_structured_vs_native",
            "adjacent_layer_change",
        ),
    )
    rows: list[dict[str, Any]] = []
    for model, group in association.groupby("model"):
        for attention_metric, cosine_metric, scope in pairs:
            if attention_metric not in group or cosine_metric not in group:
                continue
            clean = group[[attention_metric, cosine_metric]].dropna()
            if len(clean) < 4:
                continue
            rows.append(
                {
                    "model": model,
                    "attention_metric": attention_metric.replace(
                        "motif_", "pattern_"
                    ),
                    "cosine_metric": cosine_metric,
                    "association_scope": scope,
                    "spearman_rho": clean[attention_metric]
                    .rank()
                    .corr(clean[cosine_metric].rank()),
                    "n_pair_layers": len(clean),
                    "layer_alignment": "attention_l_to_hidden_l_plus_1",
                }
            )
    return pd.DataFrame(rows)


def _unit_slopes(
    frame: pd.DataFrame,
    metric: str,
    slope_name: str,
) -> pd.DataFrame:
    if frame.empty or metric not in frame:
        return pd.DataFrame(columns=["model", slope_name])
    group_columns = [
        column
        for column in (
            "model",
            "seed",
            "source_type",
            "background_type",
            "pattern_id",
            "pattern_family",
            "strength_mode",
            "strength_value",
        )
        if column in frame
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        clean = group[["length", metric]].dropna().drop_duplicates("length")
        if clean["length"].nunique() < 2:
            continue
        slope = float(
            np.polyfit(
                np.log(clean["length"].to_numpy(dtype=float)),
                clean[metric].to_numpy(dtype=float),
                deg=1,
            )[0]
        )
        key_values = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_columns, key_values)),
                slope_name: slope,
            }
        )
    return pd.DataFrame(rows)


def _peak_layers(
    frame: pd.DataFrame,
    metric: str,
    output_name: str,
) -> pd.DataFrame:
    if frame.empty or metric not in frame:
        return pd.DataFrame(columns=["model", output_name])
    valid = (
        frame.dropna(subset=[metric])
        .groupby(["model", "layer"], as_index=False)[metric]
        .mean()
    )
    if valid.empty:
        return pd.DataFrame(columns=["model", output_name])
    indices = valid.groupby("model")[metric].idxmax()
    return valid.loc[indices, ["model", "layer"]].rename(
        columns={"layer": output_name}
    )


def _primary_scores(
    context_final: pd.DataFrame,
    diffusion_final: pd.DataFrame,
    context_layers: pd.DataFrame,
    diffusion_layers: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    models = sorted(
        set(context_final.get("model", ())) | set(diffusion_final.get("model", ()))
    )
    context_slopes = _unit_slopes(
        context_final, "context_separation", "context_collapse_slope"
    )
    diffusion_slopes = _unit_slopes(
        diffusion_final, "far_field", "length_amplification_slope"
    )
    context_peak = _peak_layers(
        context_layers, "context_separation", "layer_max_contextualization"
    )
    leakage_peak = _peak_layers(
        diffusion_layers, "far_field", "layer_max_leakage"
    )
    rows = []
    for model_index, model in enumerate(models):
        rng = np.random.default_rng(seed + model_index * 1009)
        context_stats = _cluster_bootstrap_mean(
            context_final.loc[context_final["model"] == model],
            "context_separation",
            replicates,
            rng,
        )
        far_stats = _cluster_bootstrap_mean(
            diffusion_final.loc[diffusion_final["model"] == model],
            "far_field",
            replicates,
            rng,
        )
        relative_stats = _cluster_bootstrap_mean(
            diffusion_final.loc[diffusion_final["model"] == model],
            "relative_distal",
            replicates,
            rng,
        )
        extra_diffusion_stats = {
            metric: _cluster_bootstrap_mean(
                diffusion_final.loc[diffusion_final["model"] == model],
                metric,
                replicates,
                rng,
            )
            for metric in (
                "relative_distal_025",
                "relative_distal_050",
                "relative_distal_075",
            )
        }
        context_slope_stats = _cluster_bootstrap_mean(
            context_slopes.loc[context_slopes["model"] == model]
            if not context_slopes.empty
            else pd.DataFrame(),
            "context_collapse_slope",
            replicates,
            rng,
        )
        length_slope_stats = _cluster_bootstrap_mean(
            diffusion_slopes.loc[diffusion_slopes["model"] == model]
            if not diffusion_slopes.empty
            else pd.DataFrame(),
            "length_amplification_slope",
            replicates,
            rng,
        )
        context_layer = context_peak.loc[
            context_peak["model"] == model, "layer_max_contextualization"
        ]
        leakage_layer = leakage_peak.loc[
            leakage_peak["model"] == model, "layer_max_leakage"
        ]
        rows.append(
            {
                "model": model,
                "context_separation": context_stats[0],
                "context_separation_ci_low": context_stats[1],
                "context_separation_ci_high": context_stats[2],
                "context_units": context_stats[3],
                "far_field_diffusion": far_stats[0],
                "far_field_diffusion_ci_low": far_stats[1],
                "far_field_diffusion_ci_high": far_stats[2],
                "diffusion_units": far_stats[3],
                "relative_distal_diffusion": relative_stats[0],
                "relative_distal_diffusion_ci_low": relative_stats[1],
                "relative_distal_diffusion_ci_high": relative_stats[2],
                "relative_distal_units": relative_stats[3],
                **{
                    f"{metric}_mean": stats[0]
                    for metric, stats in extra_diffusion_stats.items()
                },
                **{
                    f"{metric}_ci_low": stats[1]
                    for metric, stats in extra_diffusion_stats.items()
                },
                **{
                    f"{metric}_ci_high": stats[2]
                    for metric, stats in extra_diffusion_stats.items()
                },
                **{
                    f"{metric}_units": stats[3]
                    for metric, stats in extra_diffusion_stats.items()
                },
                "length_amplification_slope": length_slope_stats[0],
                "length_amplification_ci_low": length_slope_stats[1],
                "length_amplification_ci_high": length_slope_stats[2],
                "length_slope_units": length_slope_stats[3],
                "context_collapse_slope": context_slope_stats[0],
                "context_collapse_ci_low": context_slope_stats[1],
                "context_collapse_ci_high": context_slope_stats[2],
                "context_slope_units": context_slope_stats[3],
                "layer_max_contextualization": (
                    float(context_layer.iloc[0]) if len(context_layer) else math.nan
                ),
                "layer_max_leakage": (
                    float(leakage_layer.iloc[0]) if len(leakage_layer) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _model_pairwise_tests(
    context_controls: pd.DataFrame,
    diffusion_final: pd.DataFrame,
    attention_convergence: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Compare every model pair on exactly paired full-transcript windows."""
    endpoint_frames: list[tuple[str, pd.DataFrame, str]] = []

    if not context_controls.empty:
        context = context_controls.copy()
        final_layer = context.groupby(
            ["model", "length", "pair_group_id"]
        )["layer"].transform("max")
        context = context[context["layer"] == final_layer]
        context = (
            context.groupby(
                ["model", "length", "pair_group_id", "seed"],
                as_index=False,
            )[
                [
                    "delta_context_separation_structured_vs_native",
                    "cross_context_same_base_cos_structured",
                ]
            ]
            .mean()
        )
        endpoint_frames.extend(
            (
                endpoint,
                context,
                metric,
            )
            for endpoint, metric in (
                (
                    "final_context_gain",
                    "delta_context_separation_structured_vs_native",
                ),
                (
                    "final_cross_region_same_base_cosine",
                    "cross_context_same_base_cos_structured",
                ),
            )
        )

    if not diffusion_final.empty:
        endpoint_frames.extend(
            [
                ("final_relative_distal_diffusion", diffusion_final, "relative_distal"),
                ("final_relative_distal_025", diffusion_final, "relative_distal_025"),
                ("final_relative_distal_050", diffusion_final, "relative_distal_050"),
                ("final_relative_distal_075", diffusion_final, "relative_distal_075"),
                ("final_far_field_diffusion", diffusion_final, "far_field"),
            ]
        )
    if not attention_convergence.empty:
        endpoint_frames.append(
            (
                "late_layer_attention_convergence",
                attention_convergence,
                "normalized_mean_distance_late_minus_middle",
            )
        )

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 104729)
    for endpoint, frame, metric in endpoint_frames:
        required = {"model", "length", "pair_group_id", metric}
        if frame.empty or not required.issubset(frame):
            continue
        grouped = (
            frame.groupby(
                ["model", "length", "pair_group_id"],
                as_index=False,
            )[metric]
            .mean()
        )
        for length, length_frame in grouped.groupby("length"):
            wide = length_frame.pivot(
                index="pair_group_id", columns="model", values=metric
            )
            for left_model, right_model in itertools.combinations(
                sorted(wide.columns), 2
            ):
                paired = wide[[left_model, right_model]].dropna()
                if paired.empty:
                    continue
                differences = (
                    paired[right_model] - paired[left_model]
                ).to_numpy(dtype=float)
                stats = _paired_bootstrap(
                    differences, replicates, rng
                )
                standard_deviation = (
                    float(np.std(differences, ddof=1))
                    if len(differences) > 1
                    else math.nan
                )
                left_label = display_model_name(left_model)
                right_label = display_model_name(right_model)
                rows.append(
                    {
                        "endpoint": endpoint,
                        "length": int(length),
                        "model_a": left_model,
                        "model_b": right_model,
                        "comparison": (
                            f"{right_label}_minus_{left_label}"
                        ),
                        "mean_difference": stats[0],
                        "ci_low": stats[1],
                        "ci_high": stats[2],
                        "p_value": stats[3],
                        "paired_windows": stats[4],
                        "cohen_dz": (
                            stats[0] / standard_deviation
                            if math.isfinite(standard_deviation)
                            and standard_deviation > 0
                            else math.nan
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = _assign_grouped_q_values(result)
    return result


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    digits: int = 5,
    headers: Mapping[str, str] | None = None,
) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "_No estimable rows._"
    labels = [
        (headers or {}).get(column, column) for column in available
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in available) + "|",
    ]
    for _, row in frame[available].iterrows():
        values = []
        for column, value in zip(available, row):
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    values.append("")
                elif column in {"length", "n_pairs", "paired_windows"}:
                    values.append(str(int(value)))
                else:
                    values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    profile: str,
    length_primary: pd.DataFrame,
    diagnostic_scores: pd.DataFrame,
    correlations: pd.DataFrame,
    paired_tests: pd.DataFrame,
    context_convergence: pd.DataFrame,
    attention_convergence: pd.DataFrame,
    cosine_plot_count: int,
    attention_plot_count: int,
) -> None:
    length_columns = (
        "model",
        "length",
        "n_pairs",
        "delta_cs",
        "delta_cs_ci_low",
        "delta_cs_ci_high",
        "c_cross",
        "c_cross_ci_low",
        "c_cross_ci_high",
        "d_distal",
        "d_distal_ci_low",
        "d_distal_ci_high",
    )
    table_10240 = (
        length_primary[
            pd.to_numeric(length_primary["length"], errors="coerce") == 10240
        ]
        if "length" in length_primary
        else pd.DataFrame()
    )
    primary_10240_table = _markdown_table(
        table_10240,
        length_columns,
        digits=6,
        headers=LENGTH_PRIMARY_HEADERS,
    )
    length_sweep_table = _markdown_table(
        length_primary,
        length_columns,
        digits=6,
        headers=LENGTH_PRIMARY_HEADERS,
    )
    diagnostic_table = _markdown_table(
        _order_models_in_frame(diagnostic_scores),
        (
            "model",
            "context_separation",
            "context_separation_ci_low",
            "context_separation_ci_high",
            "far_field_diffusion",
            "far_field_diffusion_ci_low",
            "far_field_diffusion_ci_high",
            "relative_distal_025_mean",
            "relative_distal_050_mean",
            "relative_distal_075_mean",
            "length_amplification_slope",
            "context_collapse_slope",
            "layer_max_contextualization",
            "layer_max_leakage",
        ),
        headers=DIAGNOSTIC_HEADERS,
    )
    correlation_table = _markdown_table(
        _order_models_in_frame(correlations),
        (
            "model",
            "attention_metric",
            "cosine_metric",
            "association_scope",
            "spearman_rho",
            "n_pair_layers",
        ),
    )
    paired_columns = (
        "endpoint",
        "length",
        "model_a",
        "model_b",
        "comparison",
        "mean_difference",
        "ci_low",
        "ci_high",
        "cohen_dz",
        "p_value",
        "q_value_bh",
        "paired_windows",
    )
    primary_paired_table = _markdown_table(
        _paired_tests_for_report(paired_tests, PRIMARY_BH_ENDPOINTS),
        paired_columns,
    )
    sensitivity_paired_table = _markdown_table(
        _paired_tests_for_report(paired_tests, SENSITIVITY_PAIRED_ENDPOINTS),
        paired_columns,
    )
    context_columns = [
        column
        for column in (
            "delta_context_separation_structured_vs_native_late_minus_middle",
            "context_separation_structured_late_minus_middle",
            "cross_context_same_base_cos_structured_late_minus_middle",
        )
        if column in context_convergence
    ]
    context_summary = (
        _order_models_in_frame(
            context_convergence.groupby("model", as_index=False)[
                context_columns
            ].mean()
        )
        if not context_convergence.empty and context_columns
        else pd.DataFrame()
    )
    context_convergence_table = _markdown_table(
        context_summary,
        ("model", *context_columns),
        headers=CONVERGENCE_HEADERS,
    )
    attention_columns = [
        column
        for column in (
            "normalized_mean_distance_late_minus_middle",
            "mass_gt_1024_late_minus_middle",
            "mass_gt_2048_late_minus_middle",
        )
        if column in attention_convergence
    ]
    attention_summary = (
        _order_models_in_frame(
            attention_convergence.groupby("model", as_index=False)[
                attention_columns
            ].mean()
        )
        if not attention_convergence.empty and attention_columns
        else pd.DataFrame()
    )
    attention_convergence_table = _markdown_table(
        attention_summary, ("model", *attention_columns)
    )
    content = f"""# Long-Context Representation Benchmark Report

- Profile: `{profile}`
- Generated: `{datetime.now(timezone.utc).isoformat()}`
- Statistical unit: transcript-pair. Each length group uses 10 complete-mRNA
  pairs. Metrics are first averaged within a pair, then reported as the mean
  across pairs. Confidence intervals are 95% bootstrap intervals over those
  pair-level means.

## Benchmark protocol

The benchmark uses complete mRNAs in length groups of 1,024, 2,048, 4,096,
8,192, and 10,240 nt. For a transcript of length `L`, a centered interval of
width `W = round(L/32)` is reordered while preserving nucleotide composition;
all positions outside the interval remain unchanged. The unmodified native
transcript is the control. An 8-nt buffer around the intervention interval is
excluded from the background set `B`.

## Primary cosine endpoints

These are the three final-layer measures used in the technical report:
Additional Context Separation (`ΔCS` ↑), Cross-region Same-base Similarity
(`C_cross` ↓), and Relative-distal Diffusion (`D_distal`). They should be
interpreted jointly.

### Final-layer metrics at 10,240 nt

{primary_10240_table}

### Length sweep

{length_sweep_table}

### Definitions

**Context Separation (CS).** For each nucleotide `b`, CS is the
pair-count-weighted same-region cosine minus the cross-region same-base
cosine:

`CS^(b) = C_same^(b) − C_IB^(b)`,

where `C_same` combines within-interval and within-background pairs, and
`C_IB` averages pairs that span the intervention interval `I` and background
`B`.

**Additional Context Separation (ΔCS).** At the final layer, `ΔCS` is the
mean over bases of `CS_structured − CS_native`. Larger positive values
indicate a greater increase in regional separation after the
composition-preserving rearrangement.

**Cross-region Same-base Similarity (C_cross).** Mean over bases of the
structured-sequence cross-region same-base cosine `C_IB`. Lower values
indicate stronger regional separation of same-nucleotide representations.

**Relative-distal Diffusion (D_distal).** For unchanged positions `i ∉ I`,
`D_i = 1 − cos(h_i,structured, h_i,native)` and `r_i = d_i / d_max`.
`D_distal` averages final-layer `D_i` over positions with `r_i ≥ 0.75`
(the most distal 25%). Lower values indicate weaker propagation into distant
unchanged regions.

## Paired model comparisons

Differences are `model_b - model_a` on the same transcript pairs. Confidence
intervals use paired bootstrap. `q_value_bh` is Benjamini–Hochberg FDR
computed **separately within each primary endpoint family**: `ΔCS`
(`final_context_gain`), `C_cross`
(`final_cross_region_same_base_cosine`), and `D_distal`
(`final_relative_distal_075`). Attention uses the same procedure inside
`late_layer_attention_convergence` only.

### Primary endpoints

{primary_paired_table}

### Sensitivity endpoints

Far-field diffusion uses an absolute distance threshold (`≥1024 nt`) and is
undefined when no such positions exist. Relative-distal `r ≥ 0.25` and
`r ≥ 0.50` are threshold-sensitivity checks. The alias
`final_relative_distal_diffusion` equals `D_distal` under the configured
primary threshold `0.75` and is not BH-adjusted.

{sensitivity_paired_table}

## Layer-wise structured-pattern specificity and convergence

Late-minus-middle `ΔCS` is the change in Additional Context Separation from
the middle quarter of layers to the final quarter. Negative values indicate
loss of rearrangement-induced separation in late layers.

{context_convergence_table}

Attention block `l` is aligned to hidden state `l+1`. Negative attention
late-minus-middle distance indicates late-layer selective re-localization;
positive values indicate continued distance expansion.

{attention_convergence_table}

## Supplementary cosine diagnostics

The following scores are retained for diagnostics. They are **not** the
technical-report primary endpoints. `CS_structured` is final-layer Context
Separation on the rearranged sequence only (not `ΔCS`) and is pooled across
lengths. Far-field diffusion uses absolute distance `≥1024 nt`. Length-pooled
`D_distal` equals the `r ≥ 0.75` column. Layer-peak columns record the layer
of maximum `CS_structured` and maximum far-field diffusion.

{diagnostic_table}

Optional anisotropy-adjusted cosines subtract a scalar same-base baseline
rather than whitening. Because the same scalar is subtracted from both CS
terms, adjusted CS equals raw CS; adjusted component cosines remain available
for diagnosing layer anisotropy.

## Cosine diagnostic figures

Generated {cosine_plot_count} cosine artifacts under `plots/cosine/`:
same-base context geometry, joint `ΔCS`–`D_distal` trajectories, centroid
margins when enabled, attention–similarity coupling, length-scaling overview,
and relative-distance threshold sensitivity.

## Attention-cosine association

{correlation_table}

Attention endpoints are secondary mechanistic diagnostics. A correlation does
not by itself establish that attention caused the hidden-state change.
Layer associations use the explicit `attention_l -> hidden_l+1` alignment.

## Attention diagnostic figures

Generated {attention_plot_count} aggregate attention artifacts under
`plots/attention/`: layer-wise raw/normalized distance, distance-bin mass,
and late-layer convergence.

"""
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_replicates < 2:
        raise ValueError("--bootstrap-replicates must be at least 2")
    config_path = _resolve(args.config, PROJECT_ROOT)
    experiment = load_experiment_config(config_path, profile=args.profile)
    result_root = PROJECT_ROOT / "outputs" / experiment.profile
    cosine_dir = _resolve(args.cosine_dir or result_root / "cosine", PROJECT_ROOT)
    attention_dir = _resolve(
        args.attention_dir or result_root / "attention", PROJECT_ROOT
    )
    output_dir = _resolve(args.output or result_root / "analysis", PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _use_length_groups(
        _read_csv(cosine_dir / "cosine_context.csv", required=True)
    )
    diffusion = _use_length_groups(
        _read_csv(cosine_dir / "cosine_diffusion.csv", required=True)
    )
    attention_summary = _read_csv(
        attention_dir / "attention_summary.csv", required=False
    )
    attention_summary = _use_length_groups(attention_summary)
    trajectories = _use_length_groups(
        _read_csv(
            cosine_dir / "cosine_token_trajectories.csv", required=False
        )
    )

    context_final = _final_context(context)
    diffusion_final = _final_diffusion(diffusion)
    context_controls = _context_control_comparison(context)
    context_dynamics = _context_layer_dynamics(context_controls)
    context_convergence = _late_layer_summary(context_dynamics)
    attention_dynamics = _attention_layer_dynamics(attention_summary)
    attention_convergence = _late_layer_summary(
        attention_dynamics, layer_column="hidden_layer"
    )
    attention_similarity = _attention_similarity_association(
        attention_dynamics, context_dynamics
    )
    context_layers = _numeric(
        context[
            (context["control_type"] == "structured")
            & (context["metric_variant"] == "raw")
        ],
        ("layer", "context_separation", "length"),
    )
    context_layers = (
        context_layers.groupby(
            [
                column
                for column in (
                    "model",
                    "job_id",
                    "seed",
                    "length",
                    "pattern_family",
                    "layer",
                )
                if column in context_layers
            ],
            as_index=False,
            dropna=False,
        )["context_separation"]
        .mean()
        .dropna()
    )
    diffusion_layers = _numeric(
        diffusion[
            (diffusion["control_type"].isin(["native", "none"]))
            & (diffusion["metric_variant"] == "raw")
        ].drop_duplicates(["model", "pair_group_id", "layer"]),
        ("layer", "far_field", "length"),
    )
    if "degenerate_control" in diffusion_layers:
        diffusion_layers = diffusion_layers[
            diffusion_layers["degenerate_control"].astype(str).str.lower()
            != "true"
        ]

    primary = _primary_scores(
        context_final,
        diffusion_final,
        context_layers,
        diffusion_layers,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    length_primary = _length_primary_scores(
        _pair_level_paper_metrics(context_controls, diffusion_final),
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    correlations = _association_correlations(attention_similarity)
    paired_tests = _model_pairwise_tests(
        context_controls,
        diffusion_final,
        attention_convergence,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )

    primary.to_csv(output_dir / "primary_scores.csv", index=False)
    length_primary.to_csv(
        output_dir / "length_primary_scores.csv", index=False
    )
    correlations.to_csv(
        output_dir / "attention_cosine_correlations.csv", index=False
    )
    paired_tests.to_csv(
        output_dir / "model_pairwise_tests.csv", index=False
    )
    context_final.to_csv(output_dir / "context_final_layer_units.csv", index=False)
    diffusion_final.to_csv(
        output_dir / "diffusion_final_layer_units.csv", index=False
    )
    context_controls.to_csv(
        output_dir / "context_control_comparison.csv", index=False
    )
    context_dynamics.to_csv(
        output_dir / "context_layer_dynamics.csv", index=False
    )
    context_convergence.to_csv(
        output_dir / "context_late_layer_convergence.csv", index=False
    )
    attention_dynamics.to_csv(
        output_dir / "attention_layer_dynamics.csv", index=False
    )
    attention_convergence.to_csv(
        output_dir / "attention_late_layer_convergence.csv", index=False
    )
    attention_similarity.to_csv(
        output_dir / "attention_similarity_layer_association.csv", index=False
    )

    if args.no_plots:
        plot_paths: list[str] = []
    else:
        plot_paths = plot_cosine_diagnostics(
            context,
            diffusion,
            output_dir / "plots" / "cosine",
            context_controls=context_controls,
            trajectories=trajectories,
            attention_similarity=attention_similarity,
        )
        plot_paths.extend(
            plot_attention_diagnostics(
                attention_summary,
                output_dir / "plots" / "attention",
            )
        )
    _write_report(
        output_dir / "report.md",
        experiment.profile,
        length_primary,
        primary,
        correlations,
        paired_tests,
        context_convergence,
        attention_convergence,
        sum("/plots/cosine/" in path for path in plot_paths),
        sum("/plots/attention/" in path for path in plot_paths),
    )
    metadata = {
        "schema_version": 2,
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": experiment.profile,
        "config": public_path(experiment.config_path, PROJECT_ROOT),
        "cosine_dir": public_path(cosine_dir, PROJECT_ROOT),
        "attention_dir": public_path(attention_dir, PROJECT_ROOT),
        "output_dir": public_path(output_dir, PROJECT_ROOT),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "primary_models": len(primary),
        "length_primary_rows": len(length_primary),
        "attention_correlations": len(correlations),
        "model_pairwise_tests": len(paired_tests),
        "context_control_rows": len(context_controls),
        "context_dynamics_rows": len(context_dynamics),
        "attention_dynamics_rows": len(attention_dynamics),
        "attention_similarity_rows": len(attention_similarity),
        "token_trajectory_rows": len(trajectories),
        "attention_cosine_layer_alignment": "attention_l_to_hidden_l_plus_1",
        "plots": plot_paths,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    metadata = run(parse_args(argv))
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

