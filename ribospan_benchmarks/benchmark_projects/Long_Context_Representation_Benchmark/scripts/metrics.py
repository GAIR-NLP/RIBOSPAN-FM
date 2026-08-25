# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cosine probes: adjusted_cosine = raw_cosine - anisotropy baseline (not whitening)."""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch


BASES: tuple[str, ...] = ("A", "C", "G", "T")
EPSILON = 1.0e-12


@dataclass(frozen=True)
class ScalarStats:
    """Summary of a one-dimensional collection of sampled values."""

    mean: float | None
    std: float | None
    n: int


def stable_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Return row-wise cosine in float32 without constructing an L x L matrix.

    Leading dimensions are broadcast by PyTorch.  Zero-norm rows produce zero
    rather than NaN, and small floating-point excursions are clipped to [-1, 1].
    """

    left_f = left.float()
    right_f = right.float()
    numerator = (left_f * right_f).sum(dim=-1)
    denominator = left_f.norm(dim=-1) * right_f.norm(dim=-1)
    cosine = numerator / denominator.clamp_min(eps)
    cosine = torch.where(denominator > eps, cosine, torch.zeros_like(cosine))
    return cosine.clamp(min=-1.0, max=1.0)


def center_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Center a layer by this sample's real-token mean."""

    if hidden.ndim != 2 or hidden.shape[0] == 0:
        raise ValueError("hidden must have non-empty shape [tokens, dim]")
    hidden_f = hidden.float()
    return hidden_f - hidden_f.mean(dim=0, keepdim=True)


def scalar_stats(values: torch.Tensor) -> ScalarStats:
    """Summarize finite values using population standard deviation."""

    finite = values.float()[torch.isfinite(values)]
    if finite.numel() == 0:
        return ScalarStats(mean=None, std=None, n=0)
    return ScalarStats(
        mean=float(finite.mean().item()),
        std=float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
        n=int(finite.numel()),
    )


def _combination_count(n: int) -> int:
    return n * (n - 1) // 2


def _unrank_pair(rank: int, n: int) -> tuple[int, int]:
    """Map a rank in ``range(n choose 2)`` to a unique pair i < j."""

    if n < 2 or rank < 0 or rank >= _combination_count(n):
        raise ValueError(f"Invalid pair rank={rank} for n={n}")

    def pairs_before(index: int) -> int:
        return index * (2 * n - index - 1) // 2

    low, high = 0, n - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if pairs_before(middle) <= rank:
            low = middle
        else:
            high = middle
    first = low
    second = first + 1 + rank - pairs_before(first)
    return first, second


def _sample_within_pairs(
    indices: Sequence[int],
    max_pairs: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if max_pairs <= 0 or len(indices) < 2:
        return []
    total = _combination_count(len(indices))
    ranks: Iterable[int]
    if total <= max_pairs:
        ranks = range(total)
    else:
        ranks = rng.sample(range(total), max_pairs)
    return [
        (indices[first], indices[second])
        for first, second in (_unrank_pair(rank, len(indices)) for rank in ranks)
    ]


def _sample_cross_pairs(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    max_pairs: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if max_pairs <= 0 or not left_indices or not right_indices:
        return []
    total = len(left_indices) * len(right_indices)
    ranks: Iterable[int]
    if total <= max_pairs:
        ranks = range(total)
    else:
        ranks = rng.sample(range(total), max_pairs)
    right_size = len(right_indices)
    return [
        (left_indices[rank // right_size], right_indices[rank % right_size])
        for rank in ranks
    ]


def _sample_positions(
    indices: Sequence[int],
    max_positions: int,
    rng: random.Random,
) -> list[int]:
    if max_positions <= 0 or len(indices) <= max_positions:
        return list(indices)
    return sorted(rng.sample(list(indices), max_positions))


def _cosines_for_pairs(
    hidden: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    if not pairs:
        return torch.empty(0, dtype=torch.float32)
    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    return stable_cosine(hidden.index_select(0, left), hidden.index_select(0, right))


def _cross_cosines(
    hidden: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    return _cosines_for_pairs(hidden, pairs)


def _weighted_mean(
    values: Sequence[tuple[float | None, int]],
) -> float | None:
    usable = [(value, count) for value, count in values if value is not None and count > 0]
    if not usable:
        return None
    total = sum(count for _, count in usable)
    return sum(float(value) * count for value, count in usable) / total


def estimate_anisotropy_baseline(
    hidden: torch.Tensor,
    labels: Sequence[str] | None = None,
    *,
    max_pairs: int = 4096,
    seed: int = 0,
    match_labels: bool = True,
) -> ScalarStats:
    """Estimate a same-layer scalar baseline from random distinct token pairs.

    When ``match_labels`` is true, random pairs are constrained to have the same
    nucleotide label.  Sampling is uniform over all eligible unordered pairs,
    not over labels.  No token-token similarity matrix is materialized.
    """

    if hidden.ndim != 2:
        raise ValueError(f"hidden must have shape [tokens, dim], got {tuple(hidden.shape)}")
    token_count = int(hidden.shape[0])
    if labels is not None and len(labels) != token_count:
        raise ValueError("labels length must equal hidden token count")
    rng = random.Random(seed)

    if not match_labels or labels is None:
        pairs = _sample_within_pairs(list(range(token_count)), max_pairs, rng)
        return scalar_stats(_cosines_for_pairs(hidden, pairs))

    groups: list[list[int]] = []
    for label in sorted(set(labels)):
        group = [index for index, value in enumerate(labels) if value == label]
        if len(group) >= 2:
            groups.append(group)
    counts = [_combination_count(len(group)) for group in groups]
    total = sum(counts)
    if total == 0 or max_pairs <= 0:
        return ScalarStats(mean=None, std=None, n=0)

    sampled_ranks: Iterable[int]
    if total <= max_pairs:
        sampled_ranks = range(total)
    else:
        sampled_ranks = rng.sample(range(total), max_pairs)
    cumulative: list[int] = []
    running = 0
    for count in counts:
        running += count
        cumulative.append(running)

    pairs: list[tuple[int, int]] = []
    for global_rank in sampled_ranks:
        group_index = bisect.bisect_right(cumulative, global_rank)
        previous = cumulative[group_index - 1] if group_index else 0
        first, second = _unrank_pair(global_rank - previous, len(groups[group_index]))
        pairs.append((groups[group_index][first], groups[group_index][second]))
    return scalar_stats(_cosines_for_pairs(hidden, pairs))


def _metric_row_values(
    raw_cross: ScalarStats,
    raw_motif: ScalarStats,
    raw_background: ScalarStats,
    baseline: float,
    variant: str,
) -> dict[str, float | None]:
    if variant == "raw":
        shift = 0.0
    elif variant == "adjusted":
        shift = baseline
    else:
        raise ValueError(f"Unsupported metric variant: {variant}")

    cross = None if raw_cross.mean is None else raw_cross.mean - shift
    motif = None if raw_motif.mean is None else raw_motif.mean - shift
    background = (
        None if raw_background.mean is None else raw_background.mean - shift
    )
    same_context = _weighted_mean(
        [(motif, raw_motif.n), (background, raw_background.n)]
    )
    separation = (
        None
        if same_context is None or cross is None
        else same_context - cross
    )
    return {
        "cross_context_same_base_cos": cross,
        "within_motif_same_base_cos": motif,
        "within_background_same_base_cos": background,
        "same_context_baseline_cos": same_context,
        "context_separation": separation,
    }


def context_cosine_rows(
    hidden_states: Sequence[torch.Tensor],
    sequence: str,
    motif_start: int,
    motif_end: int,
    *,
    max_positions_per_context: int = 512,
    max_pairs_per_metric: int = 4096,
    anisotropy_pairs: int = 4096,
    anisotropy_match_base: bool = True,
    motif_interior_trim: int = 0,
    background_exclusion_radius: int = 0,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Compute per-layer, per-base same-base contextualization rows.

    The current sample's mean over all real RNA tokens is used for the centered
    variant.  Pair samples estimate within-job quantities only; downstream
    biological inference must use job-level summaries, not token pairs as
    independent replicates.
    """

    length = len(sequence)
    if not (0 <= motif_start < motif_end <= length):
        raise ValueError(
            f"Invalid motif interval [{motif_start}, {motif_end}) for length {length}"
        )
    if motif_interior_trim < 0:
        raise ValueError("motif_interior_trim must be non-negative")
    interior_start = motif_start + motif_interior_trim
    interior_end = motif_end - motif_interior_trim
    if interior_start >= interior_end:
        raise ValueError("motif_interior_trim removes the entire motif")
    if background_exclusion_radius < 0:
        raise ValueError("background_exclusion_radius must be non-negative")
    if max_positions_per_context < 1:
        raise ValueError("max_positions_per_context must be positive")
    if max_pairs_per_metric < 1:
        raise ValueError("max_pairs_per_metric must be positive")
    if anisotropy_pairs < 1:
        raise ValueError("anisotropy_pairs must be positive")

    for hidden in hidden_states:
        if hidden.ndim != 2 or hidden.shape[0] != length:
            raise ValueError(
                "Every hidden state must have shape [len(sequence), hidden_dim]"
            )

    motif_indices = list(range(interior_start, interior_end))
    excluded_start = max(0, motif_start - background_exclusion_radius)
    excluded_end = min(length, motif_end + background_exclusion_radius)
    background_indices = list(range(0, excluded_start)) + list(
        range(excluded_end, length)
    )

    # Position and pair choices are fixed across layers and metric variants.
    pair_sets: dict[str, dict[str, Any]] = {}
    for base_offset, base in enumerate(BASES):
        base_rng = random.Random(seed + 10_000 * (base_offset + 1))
        motif_for_base = [index for index in motif_indices if sequence[index] == base]
        background_for_base = [
            index for index in background_indices if sequence[index] == base
        ]
        motif_sample = _sample_positions(
            motif_for_base, max_positions_per_context, base_rng
        )
        background_sample = _sample_positions(
            background_for_base, max_positions_per_context, base_rng
        )
        pair_sets[base] = {
            "motif_positions": motif_sample,
            "background_positions": background_sample,
            "cross": _sample_cross_pairs(
                motif_sample,
                background_sample,
                max_pairs_per_metric,
                base_rng,
            ),
            "within_motif": _sample_within_pairs(
                motif_sample, max_pairs_per_metric, base_rng
            ),
            "within_background": _sample_within_pairs(
                background_sample, max_pairs_per_metric, base_rng
            ),
        }

    rows: list[dict[str, Any]] = []
    baselines: list[float] = []
    labels = list(sequence)
    for layer, hidden in enumerate(hidden_states):
        baseline_stats = estimate_anisotropy_baseline(
            hidden,
            labels,
            max_pairs=anisotropy_pairs,
            seed=seed + 1_000_003 * (layer + 1),
            match_labels=anisotropy_match_base,
        )
        baseline = baseline_stats.mean if baseline_stats.mean is not None else 0.0
        baselines.append(baseline)
        centered = center_hidden(hidden)

        for base in BASES:
            pair_set = pair_sets[base]
            for variant, representation in (
                ("raw", hidden),
                ("centered", centered),
            ):
                cross_stats = scalar_stats(
                    _cross_cosines(representation, pair_set["cross"])
                )
                motif_stats = scalar_stats(
                    _cosines_for_pairs(representation, pair_set["within_motif"])
                )
                background_stats = scalar_stats(
                    _cosines_for_pairs(
                        representation, pair_set["within_background"]
                    )
                )
                values = _metric_row_values(
                    cross_stats, motif_stats, background_stats, 0.0, "raw"
                )
                rows.append(
                    {
                        "layer": layer,
                        "layer_name": "embedding" if layer == 0 else f"layer_{layer}",
                        "base": base,
                        "metric_variant": variant,
                        **values,
                        "anisotropy_baseline": baseline,
                        "anisotropy_baseline_n_pairs": baseline_stats.n,
                        "n_cross_pairs": cross_stats.n,
                        "n_within_motif_pairs": motif_stats.n,
                        "n_within_background_pairs": background_stats.n,
                        "n_motif_positions": len(pair_set["motif_positions"]),
                        "n_background_positions": len(
                            pair_set["background_positions"]
                        ),
                    }
                )

            raw_cross = scalar_stats(_cross_cosines(hidden, pair_set["cross"]))
            raw_motif = scalar_stats(
                _cosines_for_pairs(hidden, pair_set["within_motif"])
            )
            raw_background = scalar_stats(
                _cosines_for_pairs(hidden, pair_set["within_background"])
            )
            adjusted_values = _metric_row_values(
                raw_cross, raw_motif, raw_background, baseline, "adjusted"
            )
            rows.append(
                {
                    "layer": layer,
                    "layer_name": "embedding" if layer == 0 else f"layer_{layer}",
                    "base": base,
                    "metric_variant": "adjusted",
                    **adjusted_values,
                    "anisotropy_baseline": baseline,
                    "anisotropy_baseline_n_pairs": baseline_stats.n,
                    "n_cross_pairs": raw_cross.n,
                    "n_within_motif_pairs": raw_motif.n,
                    "n_within_background_pairs": raw_background.n,
                    "n_motif_positions": len(pair_set["motif_positions"]),
                    "n_background_positions": len(
                        pair_set["background_positions"]
                    ),
                }
            )
    return rows, baselines


def normalize_distance_bins(
    bins: Sequence[float | int | str],
) -> tuple[float, ...]:
    """Validate distance-bin lower edges and normalize the final infinity."""

    normalized: list[float] = []
    for value in bins:
        if isinstance(value, str) and value.strip().lower() in {
            "inf",
            "+inf",
            "infinity",
        }:
            normalized.append(math.inf)
        else:
            normalized.append(float(value))
    if len(normalized) < 2 or normalized[0] != 0.0:
        raise ValueError("distance bins must begin at 0 and contain at least two edges")
    if not math.isinf(normalized[-1]):
        normalized.append(math.inf)
    if any(
        right <= left
        for left, right in zip(normalized[:-1], normalized[1:])
    ):
        raise ValueError("distance bins must be strictly increasing")
    return tuple(normalized)


def distance_to_interval(position: int, start: int, end: int) -> int:
    """Distance to nearest position in the half-open interval [start, end)."""

    if position < start:
        return start - position
    if position >= end:
        return position - end + 1
    return 0


def _distance_bin_label(lower: float, upper: float) -> str:
    upper_text = "inf" if math.isinf(upper) else str(int(upper))
    return f"[{int(lower)},{upper_text})"


def _curve_summary(
    bin_records: Sequence[Mapping[str, Any]],
    far_field_threshold: float,
) -> dict[str, float | None]:
    """Summarize a sampled distance curve with robust, documented definitions."""

    curve = [
        (
            float(record["mean_distance"]),
            float(record["mean"]),
            int(record["n"]),
        )
        for record in bin_records
        if record["mean"] is not None
        and record["mean_distance"] is not None
        and int(record["n"]) > 0
    ]
    curve.sort(key=lambda item: item[0])
    if not curve:
        return {
            "local_peak": None,
            "far_field": None,
            "leakage_auc": None,
        }

    local_candidates = [item for item in curve if item[0] < far_field_threshold]
    if not local_candidates:
        local_candidates = curve
    _, local_peak, _ = max(local_candidates, key=lambda item: item[1])

    far_count = sum(int(record.get("far_n", 0)) for record in bin_records)
    far_field = (
        sum(float(record.get("far_sum", 0.0)) for record in bin_records)
        / far_count
        if far_count
        else None
    )
    far_curve = [item for item in curve if item[0] >= far_field_threshold]
    if not far_curve:
        leakage_auc = None
    elif len(far_curve) == 1:
        leakage_auc = far_curve[0][1]
    else:
        area = 0.0
        for (x0, y0, _), (x1, y1, _) in zip(far_curve[:-1], far_curve[1:]):
            area += 0.5 * (y0 + y1) * (x1 - x0)
        span = far_curve[-1][0] - far_curve[0][0]
        leakage_auc = area / span if span > 0 else far_field

    return {
        "local_peak": local_peak,
        "far_field": far_field,
        "leakage_auc": leakage_auc,
    }


def diffusion_cosine_rows(
    motif_hidden_states: Sequence[torch.Tensor],
    control_hidden_states: Sequence[torch.Tensor],
    motif_sequence: str,
    control_sequence: str,
    motif_start: int,
    motif_end: int,
    *,
    distance_bins: Sequence[float | int | str],
    far_field_threshold: float,
    relative_distal_threshold: float = 0.75,
    relative_distal_thresholds: Sequence[float] | None = None,
    max_positions_per_bin: int = 4096,
    motif_anisotropy_baselines: Sequence[float] | None = None,
    control_anisotropy_baselines: Sequence[float] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    # Compute matched-position causal diffusion without an L x L matrix.

    if len(motif_sequence) != len(control_sequence):
        raise ValueError("Matched diffusion requires sequences of equal length")
    if max_positions_per_bin < 1:
        raise ValueError("max_positions_per_bin must be positive")
    if far_field_threshold < 0:
        raise ValueError("far_field_threshold must be non-negative")
    requested_thresholds = (
        tuple(float(value) for value in relative_distal_thresholds)
        if relative_distal_thresholds is not None
        else (float(relative_distal_threshold),)
    )
    thresholds = tuple(sorted(set(requested_thresholds)))
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("relative distal thresholds must be between 0 and 1")
    primary_threshold = float(relative_distal_threshold)
    if primary_threshold not in thresholds:
        raise ValueError("relative_distal_threshold must be included in thresholds")
    length = len(motif_sequence)
    if not (0 <= motif_start < motif_end <= length):
        raise ValueError("Invalid motif interval")
    if len(motif_hidden_states) != len(control_hidden_states):
        raise ValueError("Motif and control must have the same number of layers")
    for hidden in [*motif_hidden_states, *control_hidden_states]:
        if hidden.ndim != 2 or hidden.shape[0] != length:
            raise ValueError("Hidden states must align one-to-one with sequence positions")

    edges = normalize_distance_bins(distance_bins)
    all_distances = [
        distance_to_interval(position, motif_start, motif_end)
        for position in range(length)
    ]
    matched_positions = [
        position
        for position, (motif_base, control_base) in enumerate(
            zip(motif_sequence, control_sequence)
        )
        if motif_base == control_base
    ]
    changed_total = length - len(matched_positions)
    changed_motif = sum(
        motif_sequence[position] != control_sequence[position]
        for position in range(motif_start, motif_end)
    )
    maximum_distance = max(all_distances)

    def threshold_label(value: float) -> str:
        return f"{int(round(value * 100)):03d}"

    relative_subsets: dict[float, dict[str, Any]] = {}
    for threshold in thresholds:
        eligible = [
            position
            for position in matched_positions
            if maximum_distance > 0
            and (all_distances[position] / maximum_distance) >= threshold
        ]
        offset = (
            0
            if math.isclose(threshold, 0.75)
            else 1009 * int(round(threshold * 1000))
        )
        positions = _sample_positions(
            eligible,
            max_positions_per_bin,
            random.Random(seed + 1_000_003 + offset),
        )
        relative_subsets[threshold] = {
            "positions": positions,
            "index": torch.tensor(positions, dtype=torch.long)
            if positions
            else torch.empty(0, dtype=torch.long),
            "n_total": len(eligible),
        }

    sampled_by_bin: list[dict[str, Any]] = []
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        eligible = [
            position
            for position in matched_positions
            if lower <= all_distances[position] < upper
        ]
        sampled = _sample_positions(
            eligible,
            max_positions_per_bin,
            random.Random(seed + 104_729 * (bin_index + 1)),
        )
        sampled_by_bin.append(
            {
                "index": bin_index,
                "lower": lower,
                "upper": upper,
                "positions": sampled,
                "n_total": len(eligible),
                "distances": [all_distances[position] for position in sampled],
            }
        )

    motif_baselines = list(motif_anisotropy_baselines or [])
    control_baselines = list(control_anisotropy_baselines or [])
    output_rows: list[dict[str, Any]] = []

    for layer, (motif_hidden, control_hidden) in enumerate(
        zip(motif_hidden_states, control_hidden_states)
    ):
        motif_centered = center_hidden(motif_hidden)
        control_centered = center_hidden(control_hidden)
        motif_baseline = (
            float(motif_baselines[layer]) if layer < len(motif_baselines) else 0.0
        )
        control_baseline = (
            float(control_baselines[layer])
            if layer < len(control_baselines)
            else 0.0
        )
        pair_baseline = 0.5 * (motif_baseline + control_baseline)

        for variant, left_hidden, right_hidden, cosine_shift in (
            ("raw", motif_hidden, control_hidden, 0.0),
            ("centered", motif_centered, control_centered, 0.0),
            ("adjusted", motif_hidden, control_hidden, pair_baseline),
        ):
            def subset_mean(subset: Mapping[str, Any]) -> float | None:
                if not subset["positions"]:
                    return None
                cosine = stable_cosine(
                    left_hidden.index_select(0, subset["index"]),
                    right_hidden.index_select(0, subset["index"]),
                )
                return scalar_stats(1.0 - (cosine - cosine_shift)).mean

            relative_values = {
                threshold: subset_mean(subset)
                for threshold, subset in relative_subsets.items()
            }
            temporary: list[dict[str, Any]] = []
            for bin_data in sampled_by_bin:
                positions = bin_data["positions"]
                if positions:
                    index = torch.tensor(positions, dtype=torch.long)
                    cosine = stable_cosine(
                        left_hidden.index_select(0, index),
                        right_hidden.index_select(0, index),
                    )
                    diffusion = 1.0 - (cosine - cosine_shift)
                else:
                    diffusion = torch.empty(0, dtype=torch.float32)
                stats = scalar_stats(diffusion)
                distances = bin_data["distances"]
                far_mask = torch.tensor(
                    [distance >= far_field_threshold for distance in distances],
                    dtype=torch.bool,
                )
                far_values = diffusion[far_mask]
                temporary.append(
                    {
                        "bin_index": bin_data["index"],
                        "distance_bin": _distance_bin_label(
                            bin_data["lower"], bin_data["upper"]
                        ),
                        "distance_min": bin_data["lower"],
                        "distance_max": None
                        if math.isinf(bin_data["upper"])
                        else bin_data["upper"],
                        "mean_distance": sum(distances) / len(distances)
                        if distances
                        else None,
                        "mean": stats.mean,
                        "std": stats.std,
                        "n": stats.n,
                        "n_positions": stats.n,
                        "n_positions_total": bin_data["n_total"],
                        "far_sum": float(far_values.sum().item())
                        if far_values.numel()
                        else 0.0,
                        "far_n": int(far_values.numel()),
                    }
                )

            primary_subset = relative_subsets[primary_threshold]
            summary: dict[str, Any] = {
                **_curve_summary(temporary, far_field_threshold),
                "relative_distal": relative_values[primary_threshold],
                "relative_distal_threshold": primary_threshold,
                "relative_distal_max_distance": maximum_distance,
                "relative_distal_n_positions": len(primary_subset["positions"]),
                "relative_distal_n_positions_total": primary_subset["n_total"],
            }
            for threshold, subset in relative_subsets.items():
                label = threshold_label(threshold)
                summary[f"relative_distal_{label}"] = relative_values[threshold]
                summary[f"relative_distal_{label}_n_positions"] = len(
                    subset["positions"]
                )
                summary[f"relative_distal_{label}_n_positions_total"] = subset[
                    "n_total"
                ]
            for record in temporary:
                output_rows.append(
                    {
                        "layer": layer,
                        "layer_name": "embedding" if layer == 0 else f"layer_{layer}",
                        "metric_variant": variant,
                        **record,
                        **summary,
                        "anisotropy_baseline_motif": motif_baseline,
                        "anisotropy_baseline_control": control_baseline,
                        "anisotropy_pair_baseline": pair_baseline,
                        "n_changed_motif_excluded": changed_motif,
                        "n_changed_total_excluded": changed_total,
                    }
                )
    return output_rows

def summarize_context_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse base-level context rows to job-level layer summaries.

    Bases with an available estimate are averaged equally.  The returned list
    keeps one record per layer and metric variant, so the job—not sampled token
    pairs—remains the unit available for downstream replication.
    """

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["layer"]), str(row["metric_variant"])), []
        ).append(row)
    summaries: list[dict[str, Any]] = []
    for (layer, variant), group in sorted(grouped.items()):
        separations = [
            float(row["context_separation"])
            for row in group
            if row.get("context_separation") is not None
        ]
        summaries.append(
            {
                "layer": layer,
                "layer_name": "embedding" if layer == 0 else f"layer_{layer}",
                "metric_variant": variant,
                "mean_context_separation_across_bases": (
                    sum(separations) / len(separations)
                    if separations
                    else None
                ),
                "n_bases_with_estimate": len(separations),
            }
        )
    return summaries


def sample_token_trajectories(
    hidden_states: Sequence[torch.Tensor],
    sequence: str,
    motif_start: int,
    motif_end: int,
    *,
    positions_per_base_region: int = 2,
    background_exclusion_radius: int = 8,
    projection_dim: int = 3,
    projection_seed: int = 1729,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return a bounded, deterministic view of token movement across layers.

    Complete hidden tensors are intentionally not persisted.  For each base,
    this function samples a small number of motif and background positions and
    records scalar trajectory diagnostics plus an optional fixed random
    projection.  The projection is useful for within-model visualization only;
    independently trained model spaces are not assumed to be aligned.
    """

    length = len(sequence)
    if not hidden_states:
        return []
    if not (0 <= motif_start < motif_end <= length):
        raise ValueError("Invalid motif interval for token trajectory")
    if positions_per_base_region < 1:
        raise ValueError("positions_per_base_region must be positive")
    if background_exclusion_radius < 0:
        raise ValueError("background_exclusion_radius must be non-negative")
    if not 0 <= projection_dim <= 3:
        raise ValueError("projection_dim must be between 0 and 3")
    hidden_dim = int(hidden_states[0].shape[1])
    for hidden in hidden_states:
        if hidden.ndim != 2 or tuple(hidden.shape) != (length, hidden_dim):
            raise ValueError(
                "Every hidden state must have shape [len(sequence), hidden_dim]"
            )

    excluded_start = max(0, motif_start - background_exclusion_radius)
    excluded_end = min(length, motif_end + background_exclusion_radius)
    motif_positions = list(range(motif_start, motif_end))
    background_positions = list(range(0, excluded_start)) + list(
        range(excluded_end, length)
    )

    selected: list[tuple[str, str, int, int]] = []
    for base_offset, base in enumerate(BASES):
        for region_offset, (region, candidates) in enumerate(
            (("pattern", motif_positions), ("background", background_positions))
        ):
            eligible = [position for position in candidates if sequence[position] == base]
            rng = random.Random(
                seed + 1_000_003 * (base_offset + 1) + 10_007 * (region_offset + 1)
            )
            sampled = _sample_positions(
                eligible, positions_per_base_region, rng
            )
            selected.extend(
                (region, base, position, rank)
                for rank, position in enumerate(sampled)
            )
    if not selected:
        return []

    projection: torch.Tensor | None = None
    if projection_dim:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(projection_seed))
        projection = torch.randn(
            hidden_dim,
            projection_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(float(hidden_dim))

    centroid_stats: dict[
        tuple[int, str, str], tuple[torch.Tensor, int]
    ] = {}
    for layer, hidden in enumerate(hidden_states):
        hidden_f = hidden.float()
        for base in BASES:
            for region, candidates in (
                ("pattern", motif_positions),
                ("background", background_positions),
            ):
                indices = [
                    position for position in candidates if sequence[position] == base
                ]
                if indices:
                    index = torch.tensor(indices, dtype=torch.long)
                    vectors = hidden_f.index_select(0, index)
                    centroid_stats[(layer, base, region)] = (
                        vectors.sum(dim=0),
                        len(indices),
                    )

    rows: list[dict[str, Any]] = []
    embedding = hidden_states[0].float()
    previous: torch.Tensor | None = None
    for layer, hidden in enumerate(hidden_states):
        hidden_f = hidden.float()
        projected = hidden_f @ projection if projection is not None else None
        for region, base, position, rank in selected:
            vector = hidden_f[position]
            structured_stats = centroid_stats.get(
                (layer, base, "pattern")
            )
            background_stats = centroid_stats.get(
                (layer, base, "background")
            )
            centroid_margin: float | None = None
            if structured_stats is not None and background_stats is not None:
                structured_sum, structured_count = structured_stats
                background_sum, background_count = background_stats
                if region == "pattern":
                    if structured_count <= 1:
                        continue
                    structured_centroid = (
                        structured_sum - vector
                    ) / (structured_count - 1)
                    background_centroid = (
                        background_sum / background_count
                    )
                else:
                    if background_count <= 1:
                        continue
                    structured_centroid = (
                        structured_sum / structured_count
                    )
                    background_centroid = (
                        background_sum - vector
                    ) / (background_count - 1)
                structured_cos = stable_cosine(
                    vector.unsqueeze(0), structured_centroid.unsqueeze(0)
                )[0]
                background_cos = stable_cosine(
                    vector.unsqueeze(0), background_centroid.unsqueeze(0)
                )[0]
                centroid_margin = float(
                    (structured_cos - background_cos).item()
                )
            row: dict[str, Any] = {
                "layer": layer,
                "layer_name": "embedding" if layer == 0 else f"layer_{layer}",
                "position": position,
                "position_rank": rank,
                "region": region,
                "base": base,
                "distance_to_motif": distance_to_interval(
                    position, motif_start, motif_end
                ),
                "l2_norm": float(vector.norm().item()),
                "cos_to_embedding": float(
                    stable_cosine(
                        vector.unsqueeze(0), embedding[position].unsqueeze(0)
                    )[0].item()
                ),
                "step_cosine_distance": (
                    None
                    if previous is None
                    else float(
                        (
                            1.0
                            - stable_cosine(
                                vector.unsqueeze(0),
                                previous[position].unsqueeze(0),
                            )[0]
                        ).item()
                    )
                ),
                "centroid_margin": centroid_margin,
                "projection_1": None,
                "projection_2": None,
                "projection_3": None,
            }
            if projected is not None:
                for dimension in range(projection_dim):
                    row[f"projection_{dimension + 1}"] = float(
                        projected[position, dimension].item()
                    )
            rows.append(row)
        previous = hidden_f
    return rows
