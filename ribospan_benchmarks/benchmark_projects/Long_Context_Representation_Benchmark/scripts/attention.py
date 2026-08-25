# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""RNABert attention diagnostics.

Forward stays on native SDPA. A side channel reconstructs selected Q/K rows
and keeps only requested heatmap windows.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_experiment_config
from .model_io import (
    checkpoint_files_present,
    configure_runtime_rope_scaling,
    hf_encoder,
    infer_hf_backend,
    load_hf_mlm_checkpoint,
    normalize_rope_scaling_policy,
    runtime_rope_scaling_factor,
)
from .model_src import ensure_import_path
from .plotting.export import save_figure_by_format


BIN_SPECS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("bin_0_8", 0, 8),
    ("bin_9_32", 9, 32),
    ("bin_33_128", 33, 128),
    ("bin_129_512", 129, 512),
    ("bin_513_1024", 513, 1024),
    ("bin_1025_2048", 1025, 2048),
    ("bin_2049_4096", 2049, 4096),
    ("bin_4097_plus", 4097, None),
)

METRIC_FIELDS: Tuple[str, ...] = (
    "mean_distance",
    "normalized_mean_distance",
    *(name for name, _, _ in BIN_SPECS),
    "attention_entropy",
    "effective_attended_tokens",
    "real_key_mass",
    "motif_to_motif",
    "motif_to_background",
    "background_to_motif",
    "background_to_background",
    "motif_attention_enrichment",
)

SUMMARY_FIELDS: Tuple[str, ...] = (
    "model_name",
    "model_path",
    "model_context_window",
    "job_id",
    "pair_group_id",
    "seed",
    "length",
    "length_group",
    "actual_length",
    "source_type",
    "background_type",
    "pattern_id",
    "pattern_family",
    "strength_mode",
    "strength_value",
    "control_type",
    "motif_start",
    "motif_end",
    "motif_status",
    "motif_key_length",
    "layer",
    "head",
    *METRIC_FIELDS,
)

METRIC_DEFINITIONS: Dict[str, str] = {
    "row_normalization": (
        "For each real (non-CLS/SEP) query, attention on real keys is divided "
        "by that row's real-key mass. A zero-mass row remains all zero."
    ),
    "mean_distance": (
        "Mean over real queries of sum_k p(k|q, real keys) * abs(q-k), in nucleotides."
    ),
    "normalized_mean_distance": "mean_distance / max(real_sequence_length - 1, 1).",
    "distance_bins": (
        "Mean over real queries of row-normalized real-key mass whose absolute "
        "query-key offset falls in each inclusive legacy bin."
    ),
    "attention_entropy": (
        "Mean over real queries of Shannon entropy -sum_k p(k|q) ln p(k|q), natural log."
    ),
    "effective_attended_tokens": "exp(attention_entropy).",
    "real_key_mass": (
        "Mean over real queries of the unrenormalized attention mass assigned "
        "to real keys (special-token attention is excluded)."
    ),
    "motif_region_mass": (
        "Each X_to_Y metric is the mean, conditional on source query region X, "
        "of row-normalized mass assigned to key region Y. Thus the two target "
        "masses sum to one for positive-mass rows in each source region."
    ),
    "motif_attention_enrichment": (
        "background_to_motif / (motif_key_length / real_sequence_length)."
    ),
    "motif_coordinates": (
        "motif_start is zero-based and motif_end is exclusive. Missing, non-integer, "
        "out-of-range, or non-positive spans are treated as an empty motif and motif "
        "source-query metrics and enrichment are NaN; mass into absent motif keys is zero."
    ),
}

HeadSelector = Union[int, str]
SummaryKey = Tuple[str, str, int, int]


@dataclass(frozen=True)
class MotifSpan:
    """Validated zero-based, end-exclusive motif span."""

    start: int
    end: int
    status: str

    @property
    def length(self) -> int:
        """Return the number of motif keys."""

        return max(self.end - self.start, 0)

    @property
    def valid(self) -> bool:
        """Whether the original manifest coordinates were valid."""

        return self.status == "valid"


@dataclass
class HeatmapCapture:
    """A globally row-normalized, motif-centered attention window."""

    matrix: torch.Tensor
    model_name: str
    job_id: str
    pair_group_id: str
    control_type: str
    layer: int
    head: str
    sequence_length: int
    window_start: int
    window_end: int
    motif_start: int
    motif_end: int
    motif_status: str


def clean_sequence(sequence: object) -> str:
    """Canonicalize RNA to the model's DNA alphabet at single-base resolution."""

    text = str(sequence).strip().upper().replace("U", "T")
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in text)


def validate_motif_span(job: Mapping[str, Any], sequence_length: int) -> MotifSpan:
    """Validate motif coordinates without silently clipping malformed spans."""

    start_raw = job.get("motif_start")
    end_raw = job.get("motif_end")
    if start_raw is None or end_raw is None or start_raw == "" or end_raw == "":
        return MotifSpan(0, 0, "empty")
    try:
        start = int(start_raw)
        end = int(end_raw)
    except (TypeError, ValueError):
        return MotifSpan(0, 0, "invalid_non_integer")
    if end <= start:
        return MotifSpan(0, 0, "invalid_empty_or_reversed")
    if start < 0 or end > sequence_length:
        return MotifSpan(0, 0, "invalid_out_of_range")
    return MotifSpan(start, end, "valid")


def motif_centered_window(
    sequence_length: int,
    motif: MotifSpan,
    window_size: int = 512,
) -> Tuple[int, int]:
    """Return a bounded window centered on the motif, or sequence midpoint."""

    if sequence_length <= 0:
        return (0, 0)
    width = min(max(int(window_size), 1), sequence_length)
    center = (motif.start + motif.end) // 2 if motif.valid else sequence_length // 2
    start = max(0, min(center - width // 2, sequence_length - width))
    return (start, start + width)


class StreamingCSVWriter:
    """Append rows immediately while preserving a stable resume-compatible schema."""

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = Path(path)
        self.fieldnames = tuple(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
            if tuple(header) != self.fieldnames:
                raise ValueError(
                    f"Existing CSV schema does not match current schema: {self.path}"
                )
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        if not exists:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, row: Mapping[str, Any]) -> None:
        """Write and flush one row so interruption loses at most an active head."""

        self._writer.writerow({field: row.get(field, "") for field in self.fieldnames})
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file."""

        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "StreamingCSVWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_completed_keys(path: Path) -> Set[SummaryKey]:
    """Read model/job/layer/head keys already present in a summary CSV."""

    keys: Set[SummaryKey] = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                keys.add(
                    (
                        str(row["model_name"]),
                        str(row["job_id"]),
                        int(row["layer"]),
                        int(row["head"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return keys


class AttentionSummaryCollector:
    """Reduce one layer's attention immediately, without retaining full matrices."""

    def __init__(
        self,
        row_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
        query_chunk_size: int = 32,
        completed_keys: Optional[Set[SummaryKey]] = None,
    ) -> None:
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        self.row_callback = row_callback
        self.query_chunk_size = query_chunk_size
        self.completed_keys = completed_keys if completed_keys is not None else set()
        self.rows: List[Dict[str, Any]] = []
        self._job: Optional[Mapping[str, Any]] = None
        self._model_name = ""
        self._model_path = ""
        self._model_context: Any = ""
        self._length = 0
        self._motif = MotifSpan(0, 0, "empty")
        self._summary_heads: Optional[Set[int]] = None
        self._heatmap_layers: Set[int] = set()
        self._heatmap_heads: Tuple[HeadSelector, ...] = ()
        self._heatmap_window = (0, 0)
        self._captures: List[HeatmapCapture] = []
        self._stream_state: Optional[Dict[str, Any]] = None

    def begin_job(
        self,
        model_name: str,
        model_path: Union[str, Path],
        model_context_window: Any,
        job: Mapping[str, Any],
        sequence_length: int,
        summary_heads: Optional[Iterable[int]] = None,
        heatmap_layers: Iterable[int] = (),
        heatmap_heads: Iterable[HeadSelector] = (),
        heatmap_window: Optional[Tuple[int, int]] = None,
        heatmap_window_size: int = 512,
    ) -> None:
        """Set metadata and optional heatmap requests for the next batch-size-one pass."""

        if sequence_length < 1:
            raise ValueError("Attention diagnostics require a non-empty sequence")
        self._model_name = str(model_name)
        self._model_path = str(model_path)
        self._model_context = model_context_window
        self._job = job
        self._length = int(sequence_length)
        self._motif = validate_motif_span(job, self._length)
        self._summary_heads = None if summary_heads is None else set(summary_heads)
        self._heatmap_layers = set(int(layer) for layer in heatmap_layers)
        self._heatmap_heads = tuple(heatmap_heads)
        if heatmap_window is None:
            heatmap_window = motif_centered_window(
                self._length, self._motif, heatmap_window_size
            )
        start, end = heatmap_window
        if start < 0 or end <= start or end > self._length:
            raise ValueError(
                f"Invalid heatmap window [{start}, {end}) for length {self._length}"
            )
        self._heatmap_window = (int(start), int(end))
        self._captures = []

    def wants_layer(self, layer_idx: int) -> bool:
        """Return whether this layer has summary work or a heatmap request."""

        if self._job is None:
            return False
        if layer_idx in self._heatmap_layers:
            return True
        job_id = str(self._job.get("job_id", ""))
        if self._summary_heads is None:
            return True
        return any(
            (self._model_name, job_id, layer_idx, head) not in self.completed_keys
            for head in self._summary_heads
        )

    def pop_heatmaps(self) -> List[HeatmapCapture]:
        """Transfer heatmap captures to the caller."""

        captures, self._captures = self._captures, []
        return captures

    def _emit(self, row: Dict[str, Any]) -> None:
        key = (
            str(row["model_name"]),
            str(row["job_id"]),
            int(row["layer"]),
            int(row["head"]),
        )
        if key in self.completed_keys:
            return
        if self.row_callback is None:
            self.rows.append(row)
        else:
            self.row_callback(row)
        self.completed_keys.add(key)

    def record(self, layer_idx: int, attention_probs: torch.Tensor) -> None:
        """Reduce ``[1, heads, tokens, tokens]`` probabilities for one layer."""

        if self._job is None:
            raise RuntimeError("begin_job() must be called before model forward")
        if attention_probs.ndim != 4 or attention_probs.shape[0] != 1:
            raise ValueError("Collector expects attention with batch size one")
        length = self._length
        token_start, token_end = 1, 1 + length
        if attention_probs.shape[-2] < token_end or attention_probs.shape[-1] < token_end:
            raise ValueError(
                f"Attention shape {tuple(attention_probs.shape)} is too short for "
                f"{length} real tokens plus CLS"
            )

        device = attention_probs.device
        num_heads = int(attention_probs.shape[1])
        selected_heads = (
            set(range(num_heads))
            if self._summary_heads is None
            else set(self._summary_heads)
        )
        invalid_heads = sorted(head for head in selected_heads if head < 0 or head >= num_heads)
        if invalid_heads:
            raise ValueError(f"Requested heads outside [0, {num_heads - 1}]: {invalid_heads}")

        heatmap_requested = layer_idx in self._heatmap_layers
        for selector in self._heatmap_heads if heatmap_requested else ():
            if selector != "mean" and (int(selector) < 0 or int(selector) >= num_heads):
                raise ValueError(f"Invalid heatmap head: {selector}")

        distance_sum = torch.zeros(num_heads, device=device, dtype=torch.float64)
        entropy_sum = torch.zeros_like(distance_sum)
        real_mass_sum = torch.zeros_like(distance_sum)
        bin_sums = torch.zeros(
            (num_heads, len(BIN_SPECS)), device=device, dtype=torch.float64
        )
        motif_to_motif_sum = torch.zeros_like(distance_sum)
        motif_to_background_sum = torch.zeros_like(distance_sum)
        background_to_motif_sum = torch.zeros_like(distance_sum)
        background_to_background_sum = torch.zeros_like(distance_sum)
        motif_start, motif_end = self._motif.start, self._motif.end
        motif_length = self._motif.length
        background_length = length - motif_length
        key_positions = torch.arange(length, device=device, dtype=torch.float32)

        heatmap_buffers: Dict[str, torch.Tensor] = {}
        window_start, window_end = self._heatmap_window
        window_length = window_end - window_start
        if heatmap_requested:
            for selector in self._heatmap_heads:
                label = str(selector)
                heatmap_buffers[label] = torch.empty(
                    (window_length, window_length), dtype=torch.float32
                )

        for query_start in range(0, length, self.query_chunk_size):
            query_end = min(query_start + self.query_chunk_size, length)
            # Basic slicing is a view. copy=True prevents normalization from
            # mutating probabilities still owned by the model.
            raw = attention_probs[
                0,
                :,
                token_start + query_start : token_start + query_end,
                token_start:token_end,
            ]
            block = raw.to(dtype=torch.float32, copy=True)
            row_mass = block.sum(dim=-1)
            real_mass_sum += row_mass.sum(dim=-1, dtype=torch.float64)
            positive = row_mass > 0
            block.div_(row_mass.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1))
            block.mul_(positive.unsqueeze(-1))

            entropy_sum += torch.special.entr(block).sum(
                dim=(1, 2), dtype=torch.float64
            )
            query_positions = torch.arange(
                query_start, query_end, device=device, dtype=torch.float32
            )
            distances = query_positions[:, None] - key_positions[None, :]
            distances.abs_()
            distance_sum += torch.einsum("hql,ql->h", block, distances).double()
            for bin_idx, (_, lower, upper) in enumerate(BIN_SPECS):
                if upper is None:
                    bin_weights = distances.ge(lower)
                else:
                    bin_weights = distances.ge(lower) & distances.le(upper)
                bin_sums[:, bin_idx] += torch.einsum(
                    "hql,ql->h", block, bin_weights.float()
                ).double()

            if motif_length:
                motif_key_mass = block[:, :, motif_start:motif_end].sum(dim=-1)
            else:
                motif_key_mass = torch.zeros_like(row_mass)
            background_key_mass = block.sum(dim=-1) - motif_key_mass
            motif_q_start = max(query_start, motif_start)
            motif_q_end = min(query_end, motif_end)
            if motif_q_end > motif_q_start:
                local_start = motif_q_start - query_start
                local_end = motif_q_end - query_start
                motif_to_motif_sum += motif_key_mass[:, local_start:local_end].sum(
                    dim=-1, dtype=torch.float64
                )
                motif_to_background_sum += background_key_mass[
                    :, local_start:local_end
                ].sum(dim=-1, dtype=torch.float64)
            if query_start < motif_start:
                local_end = min(query_end, motif_start) - query_start
                background_to_motif_sum += motif_key_mass[:, :local_end].sum(
                    dim=-1, dtype=torch.float64
                )
                background_to_background_sum += background_key_mass[:, :local_end].sum(
                    dim=-1, dtype=torch.float64
                )
            if query_end > motif_end:
                local_start = max(query_start, motif_end) - query_start
                background_to_motif_sum += motif_key_mass[:, local_start:].sum(
                    dim=-1, dtype=torch.float64
                )
                background_to_background_sum += background_key_mass[:, local_start:].sum(
                    dim=-1, dtype=torch.float64
                )

            if heatmap_requested:
                overlap_start = max(query_start, window_start)
                overlap_end = min(query_end, window_end)
                if overlap_end > overlap_start:
                    local_start = overlap_start - query_start
                    local_end = overlap_end - query_start
                    target_start = overlap_start - window_start
                    target_end = overlap_end - window_start
                    window_block = block[
                        :, local_start:local_end, window_start:window_end
                    ]
                    for selector in self._heatmap_heads:
                        label = str(selector)
                        if selector == "mean":
                            values = window_block.mean(dim=0)
                        else:
                            values = window_block[int(selector)]
                        heatmap_buffers[label][target_start:target_end] = (
                            values.detach().cpu()
                        )

            del block, raw

        denominator = float(length)
        mean_distance = distance_sum / denominator
        normalized_distance = mean_distance / float(max(length - 1, 1))
        entropy = entropy_sum / denominator
        real_key_mass = real_mass_sum / denominator
        bin_mass = bin_sums / denominator

        nan_values = torch.full_like(distance_sum, float("nan"))
        if motif_length:
            motif_to_motif = motif_to_motif_sum / float(motif_length)
            motif_to_background = motif_to_background_sum / float(motif_length)
        else:
            motif_to_motif = nan_values
            motif_to_background = nan_values
        if background_length:
            background_to_motif = background_to_motif_sum / float(background_length)
            background_to_background = (
                background_to_background_sum / float(background_length)
            )
        else:
            background_to_motif = nan_values
            background_to_background = nan_values
        if motif_length and background_length:
            motif_enrichment = background_to_motif / (float(motif_length) / length)
        else:
            motif_enrichment = nan_values

        meta = self._summary_metadata(layer_idx)
        for head in sorted(selected_heads):
            row: Dict[str, Any] = {
                **meta,
                "head": head,
                "mean_distance": mean_distance[head].item(),
                "normalized_mean_distance": normalized_distance[head].item(),
                "attention_entropy": entropy[head].item(),
                "effective_attended_tokens": math.exp(entropy[head].item()),
                "real_key_mass": real_key_mass[head].item(),
                "motif_to_motif": motif_to_motif[head].item(),
                "motif_to_background": motif_to_background[head].item(),
                "background_to_motif": background_to_motif[head].item(),
                "background_to_background": background_to_background[head].item(),
                "motif_attention_enrichment": motif_enrichment[head].item(),
            }
            for bin_idx, (name, _, _) in enumerate(BIN_SPECS):
                row[name] = bin_mass[head, bin_idx].item()
            self._emit(row)

        if heatmap_requested:
            for selector in self._heatmap_heads:
                self._captures.append(
                    HeatmapCapture(
                        matrix=heatmap_buffers[str(selector)],
                        model_name=self._model_name,
                        job_id=str(self._job.get("job_id", "")),
                        pair_group_id=str(self._job.get("pair_group_id", "")),
                        control_type=str(self._job.get("control_type", "")),
                        layer=layer_idx,
                        head=str(selector),
                        sequence_length=length,
                        window_start=window_start,
                        window_end=window_end,
                        motif_start=self._motif.start,
                        motif_end=self._motif.end,
                        motif_status=self._motif.status,
                    )
                )

    def begin_stream(
        self,
        layer_idx: int,
        *,
        num_heads: int,
        device: torch.device,
    ) -> None:
        """Initialize numerically identical reductions for query-chunked attention."""

        if self._job is None:
            raise RuntimeError("begin_job() must be called before model forward")
        if self._stream_state is not None:
            raise RuntimeError("A streamed attention layer is already active")
        selected_heads = (
            set(range(num_heads))
            if self._summary_heads is None
            else set(self._summary_heads)
        )
        invalid_heads = sorted(
            head for head in selected_heads if head < 0 or head >= num_heads
        )
        if invalid_heads:
            raise ValueError(
                f"Requested heads outside [0, {num_heads - 1}]: {invalid_heads}"
            )
        heatmap_requested = layer_idx in self._heatmap_layers
        for selector in self._heatmap_heads if heatmap_requested else ():
            if selector != "mean" and not 0 <= int(selector) < num_heads:
                raise ValueError(f"Invalid heatmap head: {selector}")

        zeros = torch.zeros(num_heads, device=device, dtype=torch.float64)
        window_start, window_end = self._heatmap_window
        window_length = window_end - window_start
        heatmap_buffers: Dict[str, torch.Tensor] = {}
        if heatmap_requested:
            for selector in self._heatmap_heads:
                heatmap_buffers[str(selector)] = torch.empty(
                    (window_length, window_length), dtype=torch.float32
                )
        self._stream_state = {
            "layer_idx": int(layer_idx),
            "num_heads": int(num_heads),
            "selected_heads": selected_heads,
            "heatmap_requested": heatmap_requested,
            "distance_sum": zeros.clone(),
            "entropy_sum": zeros.clone(),
            "real_mass_sum": zeros.clone(),
            "bin_sums": torch.zeros(
                (num_heads, len(BIN_SPECS)),
                device=device,
                dtype=torch.float64,
            ),
            "motif_to_motif_sum": zeros.clone(),
            "motif_to_background_sum": zeros.clone(),
            "background_to_motif_sum": zeros.clone(),
            "background_to_background_sum": zeros.clone(),
            "key_positions": torch.arange(
                self._length, device=device, dtype=torch.float32
            ),
            "heatmap_buffers": heatmap_buffers,
            "real_query_count": 0,
        }

    def record_stream_chunk(
        self,
        layer_idx: int,
        attention_probs: torch.Tensor,
        *,
        query_token_start: int,
    ) -> None:
        """Reduce one full-key attention chunk and immediately release it."""

        state = self._stream_state
        if state is None or int(state["layer_idx"]) != int(layer_idx):
            raise RuntimeError("begin_stream() must precede streamed chunks")
        if attention_probs.ndim != 4 or attention_probs.shape[0] != 1:
            raise ValueError("Collector expects attention with batch size one")
        length = self._length
        token_start, token_end = 1, 1 + length
        query_token_end = query_token_start + int(attention_probs.shape[-2])
        overlap_start = max(query_token_start, token_start)
        overlap_end = min(query_token_end, token_end)
        if overlap_end <= overlap_start:
            return
        if attention_probs.shape[-1] < token_end:
            raise ValueError("Streamed attention is too short for real RNA keys")

        local_start = overlap_start - query_token_start
        local_end = overlap_end - query_token_start
        query_start = overlap_start - token_start
        query_end = overlap_end - token_start
        raw = attention_probs[
            0, :, local_start:local_end, token_start:token_end
        ]
        block = raw.to(dtype=torch.float32, copy=True)
        row_mass = block.sum(dim=-1)
        state["real_mass_sum"] += row_mass.sum(dim=-1, dtype=torch.float64)
        positive = row_mass > 0
        block.div_(
            row_mass.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        )
        block.mul_(positive.unsqueeze(-1))
        state["entropy_sum"] += torch.special.entr(block).sum(
            dim=(1, 2), dtype=torch.float64
        )

        query_positions = torch.arange(
            query_start,
            query_end,
            device=attention_probs.device,
            dtype=torch.float32,
        )
        distances = (
            query_positions[:, None] - state["key_positions"][None, :]
        ).abs()
        state["distance_sum"] += torch.einsum(
            "hql,ql->h", block, distances
        ).double()
        for bin_idx, (_, lower, upper) in enumerate(BIN_SPECS):
            if upper is None:
                bin_weights = distances.ge(lower)
            else:
                bin_weights = distances.ge(lower) & distances.le(upper)
            state["bin_sums"][:, bin_idx] += torch.einsum(
                "hql,ql->h", block, bin_weights.float()
            ).double()

        motif_start, motif_end = self._motif.start, self._motif.end
        motif_length = self._motif.length
        motif_key_mass = (
            block[:, :, motif_start:motif_end].sum(dim=-1)
            if motif_length
            else torch.zeros_like(row_mass)
        )
        background_key_mass = block.sum(dim=-1) - motif_key_mass
        motif_q_start = max(query_start, motif_start)
        motif_q_end = min(query_end, motif_end)
        if motif_q_end > motif_q_start:
            motif_local_start = motif_q_start - query_start
            motif_local_end = motif_q_end - query_start
            state["motif_to_motif_sum"] += motif_key_mass[
                :, motif_local_start:motif_local_end
            ].sum(dim=-1, dtype=torch.float64)
            state["motif_to_background_sum"] += background_key_mass[
                :, motif_local_start:motif_local_end
            ].sum(dim=-1, dtype=torch.float64)
        if query_start < motif_start:
            background_local_end = min(query_end, motif_start) - query_start
            state["background_to_motif_sum"] += motif_key_mass[
                :, :background_local_end
            ].sum(dim=-1, dtype=torch.float64)
            state["background_to_background_sum"] += background_key_mass[
                :, :background_local_end
            ].sum(dim=-1, dtype=torch.float64)
        if query_end > motif_end:
            background_local_start = max(query_start, motif_end) - query_start
            state["background_to_motif_sum"] += motif_key_mass[
                :, background_local_start:
            ].sum(dim=-1, dtype=torch.float64)
            state["background_to_background_sum"] += background_key_mass[
                :, background_local_start:
            ].sum(dim=-1, dtype=torch.float64)

        if state["heatmap_requested"]:
            window_start, window_end = self._heatmap_window
            heatmap_overlap_start = max(query_start, window_start)
            heatmap_overlap_end = min(query_end, window_end)
            if heatmap_overlap_end > heatmap_overlap_start:
                heatmap_local_start = heatmap_overlap_start - query_start
                heatmap_local_end = heatmap_overlap_end - query_start
                target_start = heatmap_overlap_start - window_start
                target_end = heatmap_overlap_end - window_start
                window_block = block[
                    :,
                    heatmap_local_start:heatmap_local_end,
                    window_start:window_end,
                ]
                for selector in self._heatmap_heads:
                    values = (
                        window_block.mean(dim=0)
                        if selector == "mean"
                        else window_block[int(selector)]
                    )
                    state["heatmap_buffers"][str(selector)][
                        target_start:target_end
                    ] = values.detach().cpu()
        state["real_query_count"] += query_end - query_start

    def end_stream(self, layer_idx: int) -> None:
        """Finalize one streamed layer and emit the same rows/captures as ``record``."""

        state = self._stream_state
        if state is None or int(state["layer_idx"]) != int(layer_idx):
            raise RuntimeError("No matching streamed attention layer is active")
        try:
            length = self._length
            if int(state["real_query_count"]) != length:
                raise RuntimeError(
                    f"Streamed {state['real_query_count']} real queries; expected {length}"
                )
            denominator = float(length)
            mean_distance = state["distance_sum"] / denominator
            normalized_distance = mean_distance / float(max(length - 1, 1))
            entropy = state["entropy_sum"] / denominator
            real_key_mass = state["real_mass_sum"] / denominator
            bin_mass = state["bin_sums"] / denominator
            motif_length = self._motif.length
            background_length = length - motif_length
            nan_values = torch.full_like(state["distance_sum"], float("nan"))
            if motif_length:
                motif_to_motif = (
                    state["motif_to_motif_sum"] / float(motif_length)
                )
                motif_to_background = (
                    state["motif_to_background_sum"] / float(motif_length)
                )
            else:
                motif_to_motif = nan_values
                motif_to_background = nan_values
            if background_length:
                background_to_motif = (
                    state["background_to_motif_sum"] / float(background_length)
                )
                background_to_background = (
                    state["background_to_background_sum"]
                    / float(background_length)
                )
            else:
                background_to_motif = nan_values
                background_to_background = nan_values
            motif_enrichment = (
                background_to_motif / (float(motif_length) / length)
                if motif_length and background_length
                else nan_values
            )

            meta = self._summary_metadata(layer_idx)
            for head in sorted(state["selected_heads"]):
                row: Dict[str, Any] = {
                    **meta,
                    "head": head,
                    "mean_distance": mean_distance[head].item(),
                    "normalized_mean_distance": normalized_distance[head].item(),
                    "attention_entropy": entropy[head].item(),
                    "effective_attended_tokens": math.exp(entropy[head].item()),
                    "real_key_mass": real_key_mass[head].item(),
                    "motif_to_motif": motif_to_motif[head].item(),
                    "motif_to_background": motif_to_background[head].item(),
                    "background_to_motif": background_to_motif[head].item(),
                    "background_to_background": background_to_background[head].item(),
                    "motif_attention_enrichment": motif_enrichment[head].item(),
                }
                for bin_idx, (name, _, _) in enumerate(BIN_SPECS):
                    row[name] = bin_mass[head, bin_idx].item()
                self._emit(row)

            if state["heatmap_requested"]:
                window_start, window_end = self._heatmap_window
                for selector in self._heatmap_heads:
                    self._captures.append(
                        HeatmapCapture(
                            matrix=state["heatmap_buffers"][str(selector)],
                            model_name=self._model_name,
                            job_id=str(self._job.get("job_id", "")),
                            pair_group_id=str(
                                self._job.get("pair_group_id", "")
                            ),
                            control_type=str(
                                self._job.get("control_type", "")
                            ),
                            layer=layer_idx,
                            head=str(selector),
                            sequence_length=length,
                            window_start=window_start,
                            window_end=window_end,
                            motif_start=self._motif.start,
                            motif_end=self._motif.end,
                            motif_status=self._motif.status,
                        )
                    )
        finally:
            self._stream_state = None

    def _summary_metadata(self, layer_idx: int) -> Dict[str, Any]:
        assert self._job is not None
        return {
            "model_name": self._model_name,
            "model_path": self._model_path,
            "model_context_window": self._model_context,
            "job_id": self._job.get("job_id", ""),
            "pair_group_id": self._job.get("pair_group_id", ""),
            "seed": self._job.get("seed", ""),
            "length": self._job.get("length", ""),
            "length_group": self._job.get("length_group", ""),
            "actual_length": self._length,
            "source_type": self._job.get("source_type", ""),
            "background_type": self._job.get("background_type", ""),
            "pattern_id": self._job.get("pattern_id", ""),
            "pattern_family": self._job.get("pattern_family", ""),
            "strength_mode": self._job.get("strength_mode", ""),
            "strength_value": self._job.get("strength_value", ""),
            "control_type": self._job.get("control_type", ""),
            "motif_start": self._job.get("motif_start", ""),
            "motif_end": self._job.get("motif_end", ""),
            "motif_status": self._motif.status,
            "motif_key_length": self._motif.length,
            "layer": layer_idx,
        }


@contextmanager
def patched_rnabert_attention(
    model: torch.nn.Module,
    collector: AttentionSummaryCollector,
    layers: Iterable[int],
) -> Iterator[None]:
    """Temporarily patch selected self-attention layers and restore them.

    The original native-SDPA forward runs first and its outputs are returned
    untouched. A diagnostic side channel then reconstructs probabilities from
    the same Q/K projections, including RoPE, YaRN temperature, additive masks,
    dropout, and head masks. Query chunking therefore cannot alter model state.
    """

    try:
        ensure_import_path()
        if getattr(model, "ribospan", None) is not None:
            from ribospan.modeling import apply_rotary_pos_emb
        else:
            from rnabert.modeling_rnabert import apply_rotary_pos_emb
    except ImportError as exc:  # pragma: no cover - exercised in integration
        raise RuntimeError(
            "Bundled rnabert/ribospan package is not importable"
        ) from exc

    requested = set(int(layer) for layer in layers)
    encoder_layers = hf_encoder(model).encoder.layer
    invalid = sorted(layer for layer in requested if layer < 0 or layer >= len(encoder_layers))
    if invalid:
        raise ValueError(f"Requested layers outside model: {invalid}")
    originals: List[Tuple[torch.nn.Module, Callable[..., Any]]] = []

    def make_forward(
        layer_idx: int,
        original_forward: Callable[..., Tuple[torch.Tensor, ...]],
    ) -> Callable[..., Tuple[torch.Tensor, ...]]:
        def patched_forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
            output_attentions: Optional[bool] = False,
            rotary_pos_emb: Any = None,
        ) -> Tuple[torch.Tensor, ...]:
            outputs = original_forward(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                rotary_pos_emb=rotary_pos_emb,
            )
            # Diagnostics are a side channel. The model context above remains
            # the untouched native SDPA result.
            if encoder_hidden_states is not None:
                return outputs

            mixed_query_layer = self.query(hidden_states)
            if past_key_value is not None:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))

            query_layer = self.transpose_for_scores(mixed_query_layer)
            if rotary_pos_emb is not None:
                cos, sin = rotary_pos_emb
                query_layer = query_layer.permute(2, 0, 1, 3).contiguous()
                key_layer = key_layer.permute(2, 0, 1, 3).contiguous()
                query_layer = apply_rotary_pos_emb(query_layer, cos, sin)
                key_layer = apply_rotary_pos_emb(key_layer, cos, sin)
                query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
                key_layer = key_layer.permute(1, 2, 0, 3).contiguous()

            final_scale = (1.0 / math.sqrt(self.attention_head_size)) / self.temperature
            query_length = int(query_layer.shape[-2])
            key_transposed = key_layer.transpose(-1, -2)
            collector.begin_stream(
                layer_idx,
                num_heads=int(query_layer.shape[1]),
                device=query_layer.device,
            )
            for query_start in range(
                0, query_length, collector.query_chunk_size
            ):
                query_end = min(
                    query_start + collector.query_chunk_size, query_length
                )
                attention_scores = (
                    torch.matmul(
                        query_layer[:, :, query_start:query_end],
                        key_transposed,
                    )
                    * final_scale
                )
                chunk_attention_mask = attention_mask
                if (
                    attention_mask is not None
                    and attention_mask.shape[-2] == query_length
                ):
                    chunk_attention_mask = attention_mask[
                        ..., query_start:query_end, :
                    ]
                if chunk_attention_mask is not None:
                    attention_scores = attention_scores + chunk_attention_mask.to(
                        attention_scores.dtype
                    )
                attention_probs = F.softmax(attention_scores, dim=-1)
                del attention_scores
                if chunk_attention_mask is not None:
                    no_prob_mask = chunk_attention_mask < -1e-5
                    attention_probs.masked_fill_(no_prob_mask, 0.0)
                attention_probs = self.dropout(attention_probs)
                if head_mask is not None:
                    attention_probs = attention_probs * head_mask
                with torch.autocast(
                    device_type=attention_probs.device.type, enabled=False
                ):
                    collector.record_stream_chunk(
                        layer_idx,
                        attention_probs,
                        query_token_start=query_start,
                    )
                del attention_probs
            collector.end_stream(layer_idx)
            return outputs

        return patched_forward

    try:
        for layer_idx in sorted(requested):
            module = encoder_layers[layer_idx].attention.self
            original_forward = module.forward
            originals.append((module, original_forward))
            module.forward = types.MethodType(
                make_forward(layer_idx, original_forward), module
            )
        yield
    finally:
        for module, original_forward in originals:
            module.forward = original_forward


def load_rnabert_model(
    model_path: Path,
    dtype: torch.dtype,
    device: torch.device,
    requested_real_tokens: int,
    backend: str = "rnabert",
) -> torch.nn.Module:
    """Load one registry checkpoint using the bundled RNABert or RiboSpan code."""

    model, _, _, _ = load_hf_mlm_checkpoint(
        model_path,
        backend=backend,
        device=device,
        dtype=dtype,
        requested_real_tokens=requested_real_tokens,
        model_name=str(model_path),
    )
    return model


def load_rnabert_tokenizer(backend: str = "rnabert") -> Any:
    """Load the bundled single-nucleotide tokenizer for one HF backend."""

    ensure_import_path()
    if backend == "ribospan":
        import ribospan as package
        from ribospan import RiboSpanTokenizer as Tokenizer
    else:
        import rnabert as package
        from rnabert import RNABertTokenizer as Tokenizer

    package_root = Path(package.__file__).resolve().parent
    return Tokenizer(str(package_root / "vocab.txt"))


def encode_sequence(
    tokenizer: Any,
    sequence: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Encode one base per token with exactly one CLS and one SEP."""

    token_ids = [tokenizer.token_to_id(base) for base in sequence]
    ids = [tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
    }


def block_pool(matrix: torch.Tensor, max_size: int = 512) -> Tuple[torch.Tensor, int]:
    """Average square blocks until each heatmap dimension is at most max_size."""

    length = int(matrix.shape[0])
    if max_size <= 0 or length <= max_size:
        return matrix, 1
    factor = math.ceil(length / max_size)
    pooled_length = math.ceil(length / factor)
    padded_length = pooled_length * factor
    padding = padded_length - length
    padded = F.pad(matrix, (0, padding, 0, padding), value=0.0)
    pooled = padded.reshape(
        pooled_length, factor, pooled_length, factor
    ).mean(dim=(1, 3))
    return pooled, factor


def safe_name(value: object) -> str:
    """Make a filesystem-safe identifier."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


def normalize_for_display(
    matrix: torch.Tensor,
    upper_quantile: float = 0.995,
) -> torch.Tensor:
    """Log-expand weak attention and robustly scale it for visualization."""

    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0, 1]")
    values = matrix.float().clamp_min(0.0)
    positive = values[values > 0]
    if positive.numel() == 0:
        return values
    transformed = torch.log1p(values / positive.median().clamp_min(1e-12))
    if transformed.numel() > 16_000_000:
        # torch.quantile rejects very large tensors. NumPy uses a partition
        # algorithm and preserves the exact requested quantile definition.
        maximum = torch.tensor(
            float(np.quantile(transformed.numpy(), upper_quantile)),
            dtype=transformed.dtype,
        ).clamp_min(1e-12)
    else:
        maximum = torch.quantile(
            transformed.flatten(), upper_quantile
        ).clamp_min(1e-12)
    return (transformed / maximum).clamp(0.0, 1.0)


def expected_attention_distance(matrix: torch.Tensor) -> torch.Tensor:
    """Return per-query expected distance without allocating a square distance matrix."""

    length = int(matrix.shape[0])
    key_positions = torch.arange(length, dtype=torch.float32)
    output = torch.empty(length, dtype=torch.float32)
    for start in range(0, length, 256):
        end = min(start + 256, length)
        query_positions = torch.arange(start, end, dtype=torch.float32)
        distances = (query_positions[:, None] - key_positions[None, :]).abs()
        output[start:end] = (matrix[start:end].float() * distances).sum(dim=-1)
    return output


def pool_distance_weighted(
    matrix: torch.Tensor,
    max_size: int,
) -> Tuple[torch.Tensor, int]:
    """Block-pool attention×distance without materializing a full distance matrix."""

    length = int(matrix.shape[0])
    if max_size <= 0 or length <= max_size:
        key_positions = torch.arange(length, dtype=torch.float32)
        weighted = torch.empty_like(matrix, dtype=torch.float32)
        for start in range(0, length, 256):
            end = min(start + 256, length)
            query_positions = torch.arange(start, end, dtype=torch.float32)
            distances = (
                query_positions[:, None] - key_positions[None, :]
            ).abs()
            weighted[start:end] = matrix[start:end].float() * distances
        return weighted, 1

    factor = math.ceil(length / max_size)
    pooled_length = math.ceil(length / factor)
    padded_length = pooled_length * factor
    key_positions = torch.arange(length, dtype=torch.float32)
    pooled = torch.zeros((pooled_length, pooled_length), dtype=torch.float32)
    for pooled_row in range(pooled_length):
        row_start = pooled_row * factor
        row_end = min(row_start + factor, length)
        query_positions = torch.arange(row_start, row_end, dtype=torch.float32)
        distances = (query_positions[:, None] - key_positions[None, :]).abs()
        column_sums = (matrix[row_start:row_end].float() * distances).sum(dim=0)
        if padded_length > length:
            column_sums = F.pad(column_sums, (0, padded_length - length))
        pooled[pooled_row] = column_sums.reshape(pooled_length, factor).sum(
            dim=1
        ) / float(factor * factor)
    return pooled, factor


def _save_figure(
    figure: Any,
    output_dir: Path,
    stem: str,
) -> Dict[str, str]:
    return save_figure_by_format(figure, output_dir, stem)


def _plot_attention_matrix(
    matrix: torch.Tensor,
    *,
    output_dir: Path,
    stem: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    window_start: int,
    window_end: int,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    figure, axis = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
    if max(matrix.shape) > 2048:
        image = axis.imshow(
            matrix.numpy(),
            cmap=cmap,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, label=colorbar_label)
    else:
        sns.heatmap(
            matrix.numpy(),
            ax=axis,
            cmap=cmap,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": colorbar_label},
        )
    axis.set_title(title)
    axis.set_xlabel("Key token position")
    axis.set_ylabel("Query token position")
    paths = _save_figure(figure, output_dir, stem)
    plt.close(figure)
    return paths


def save_heatmap(
    capture: HeatmapCapture,
    output_dir: Path,
    pool_size: int = 512,
    norm_quantile: float = 0.995,
) -> Dict[str, Any]:
    """Restore the full five-plot single-sequence attention diagnostic.

    Matplotlib is imported lazily so summary-only runs have no plotting
    dependency.
    """

    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    sns.set_theme(style="white")
    pooled_attention, pool_factor = block_pool(capture.matrix, pool_size)
    pooled_weighted, weighted_pool_factor = pool_distance_weighted(
        capture.matrix, pool_size
    )
    if weighted_pool_factor != pool_factor:
        raise RuntimeError("Attention and distance-weighted pooling disagree")
    expected_distance = expected_attention_distance(capture.matrix)
    stem = safe_name(
        f"{capture.model_name}__{capture.job_id}__layer{capture.layer}"
        f"__head{capture.head}__window{capture.window_start}-{capture.window_end}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_suffix = f" pooled by {pool_factor}" if pool_factor > 1 else ""
    title_base = (
        f"{capture.job_id} | {capture.model_name} | "
        f"input={capture.sequence_length} | layer={capture.layer} "
        f"head={capture.head} | region={capture.window_start}-"
        f"{capture.window_end}{pooled_suffix}"
    )

    artifacts: Dict[str, str] = {}
    for artifact_name, matrix, title, label, cmap in (
        (
            "attention",
            pooled_attention,
            f"Token-token attention\n{title_base}",
            "Attention probability",
            "mako",
        ),
        (
            "attention_log_norm",
            normalize_for_display(pooled_attention, norm_quantile),
            f"Token-token attention, log-normalized display\n{title_base}",
            "Log-normalized intensity",
            "mako",
        ),
        (
            "distance_weighted",
            pooled_weighted,
            f"Distance-weighted attention\n{title_base}",
            "attention * distance",
            "rocket",
        ),
        (
            "distance_weighted_log_norm",
            normalize_for_display(pooled_weighted, norm_quantile),
            f"Distance-weighted attention, log-normalized display\n{title_base}",
            "Log-normalized intensity",
            "rocket",
        ),
    ):
        paths = _plot_attention_matrix(
            matrix,
            output_dir=output_dir,
            stem=f"{stem}__{artifact_name}_heatmap",
            title=title,
            colorbar_label=label,
            cmap=cmap,
            window_start=capture.window_start,
            window_end=capture.window_end,
        )
        for format_name, path in paths.items():
            artifacts[f"{artifact_name}_{format_name}"] = path

    x = torch.arange(capture.window_start, capture.window_end)
    figure, axis = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    axis.plot(x.numpy(), expected_distance.numpy(), linewidth=1.2, color="#2A6F97")
    axis.set_title(f"Per-token expected attention distance\n{title_base}")
    axis.set_xlabel("Query token position")
    axis.set_ylabel("Expected attention distance")
    axis.grid(True, alpha=0.25)
    expected_paths = _save_figure(
        figure, output_dir, f"{stem}__token_expected_distance"
    )
    plt.close(figure)
    artifacts["token_expected_distance_pdf"] = expected_paths["pdf"]
    artifacts["token_expected_distance_svg"] = expected_paths["svg"]

    metadata: Dict[str, Any] = {
        "model_name": capture.model_name,
        "job_id": capture.job_id,
        "pair_group_id": capture.pair_group_id,
        "control_type": capture.control_type,
        "layer": capture.layer,
        "head": capture.head,
        "sequence_length": capture.sequence_length,
        "window_start": capture.window_start,
        "window_end": capture.window_end,
        "motif_start": capture.motif_start,
        "motif_end": capture.motif_end,
        "motif_status": capture.motif_status,
        "pool_factor": pool_factor,
        "norm_quantile": norm_quantile,
        "plot_count": 5,
        **artifacts,
    }
    metadata_path = output_dir / f"{stem}__metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata))
        writer.writeheader()
        writer.writerow(metadata)
    metadata["metadata_csv"] = str(metadata_path)
    return metadata


@dataclass(frozen=True)
class ModelEntry:
    """One checkpoint resolved from models.yaml."""

    name: str
    path: Path
    context_window: Any
    registry: Mapping[str, Any]
    backend: str = "rnabert"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the module CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Stream per-layer/per-head attention summaries for manifest jobs. "
            "Existing model+job+layer+head rows are resumed automatically."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML; defaults to configs/experiment.yaml.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Experiment profile; defaults to default_profile in experiment.yaml.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Override the models.yaml path resolved from experiment.yaml.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Registry model names (space- or comma-separated); default: all.",
    )
    parser.add_argument("--manifest", default=None, help="Shared JSONL job manifest.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Override models_root from models.yaml without changing registry entries.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (for example cuda:0 or cpu); default: CUDA if available.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Process at most the first N manifest jobs.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Disable all requested heatmaps.",
    )
    parser.add_argument(
        "--heatmap-jobs",
        nargs="*",
        default=None,
        help=(
            "Job IDs for representative heatmaps. Their structured/native partners "
            "in the processed manifest are included automatically."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layers for summary/heatmaps: all, integers, comma lists, or ranges (e.g. 0,8,24-31).",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=None,
        help=(
            "Heads: all, integers/ranges, and/or mean. Summary remains a per-head "
            "table; mean is only a heatmap selector."
        ),
    )
    parser.add_argument(
        "--heatmap-window",
        type=int,
        default=None,
        help="Motif-centered real-token window width (default: 512).",
    )
    parser.add_argument(
        "--heatmap-pool-size",
        type=int,
        default=None,
        help="Maximum plotted heatmap dimension (default from experiment.yaml).",
    )
    parser.add_argument(
        "--heatmap-norm-quantile",
        type=float,
        default=None,
        help="Upper quantile for log-normalized heatmap display.",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=None,
        help="Number of query rows reduced at once (default: 32).",
    )
    return parser.parse_args(argv)


def _flatten_values(values: Optional[Sequence[str]]) -> List[str]:
    flattened: List[str] = []
    for value in values or ():
        flattened.extend(part.strip() for part in str(value).split(",") if part.strip())
    return flattened


def _parse_indices(
    values: Sequence[str],
    upper_bound: int,
    label: str,
) -> Set[int]:
    tokens = _flatten_values(values)
    if not tokens or "all" in {token.lower() for token in tokens}:
        return set(range(upper_bound))
    result: Set[int] = set()
    for token in tokens:
        if token.lower() == "mean" and label == "head":
            continue
        if "-" in token:
            pieces = token.split("-", 1)
            try:
                start, end = int(pieces[0]), int(pieces[1])
            except ValueError as exc:
                raise ValueError(f"Invalid {label} range: {token}") from exc
            if end < start:
                raise ValueError(f"Reversed {label} range: {token}")
            result.update(range(start, end + 1))
        else:
            try:
                result.add(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid {label}: {token}") from exc
    invalid = sorted(index for index in result if index < 0 or index >= upper_bound)
    if invalid:
        raise ValueError(
            f"{label.capitalize()} indices outside [0, {upper_bound - 1}]: {invalid}"
        )
    return result


def _head_selections(
    values: Sequence[str],
    num_heads: int,
) -> Tuple[Set[int], Tuple[HeadSelector, ...], bool]:
    tokens = _flatten_values(values)
    lower = {token.lower() for token in tokens}
    all_requested = not tokens or "all" in lower
    mean_requested = "mean" in lower
    explicit_heads = _parse_indices(values, num_heads, "head")
    # Summary always writes every head; selectors only affect heatmaps.
    summary_heads = set(range(num_heads))
    if all_requested:
        heatmap_heads: Tuple[HeadSelector, ...] = tuple(range(num_heads))
    else:
        selectors: List[HeadSelector] = sorted(explicit_heads)
        if mean_requested:
            selectors.append("mean")
        heatmap_heads = tuple(selectors)
    return summary_heads, heatmap_heads, all_requested


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading models.yaml requires PyYAML") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Model registry must be a YAML mapping: {path}")
    return payload


def load_model_registry(
    config_path: Path,
    selected_names: Optional[Sequence[str]] = None,
    models_dir: Optional[Path] = None,
) -> List[ModelEntry]:
    """Load and resolve model entries from the registry without presets."""

    payload = _load_yaml(config_path)
    raw_models = payload.get("models")
    normalized: List[Mapping[str, Any]] = []
    if isinstance(raw_models, list):
        normalized = [entry for entry in raw_models if isinstance(entry, Mapping)]
    elif isinstance(raw_models, Mapping):
        for name, value in raw_models.items():
            if isinstance(value, Mapping):
                normalized.append({"name": name, **value})
            else:
                normalized.append({"name": name, "source_path": value})
    else:
        raise ValueError(f"'models' must be a list or mapping in {config_path}")

    requested = set(_flatten_values(selected_names))
    root_value = payload.get("models_root", "../../../model_weights")
    root = Path(models_dir) if models_dir is not None else Path(str(root_value))
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()

    entries: List[ModelEntry] = []
    seen: Set[str] = set()
    skipped_unsupported: Set[str] = set()
    attention_backends = {"rnabert", "ribospan"}
    for raw in normalized:
        name = str(raw.get("name") or raw.get("id") or "").strip()
        if not name:
            raise ValueError("Every models.yaml entry needs a name")
        if name in seen:
            raise ValueError(f"Duplicate registry model name: {name}")
        seen.add(name)
        if requested and name not in requested:
            continue
        source = raw.get("source_path", raw.get("path", name))
        model_path = Path(str(source))
        if not model_path.is_absolute():
            model_path = root / model_path
        model_path = model_path.resolve()
        declared = raw.get("backend")
        if declared is None or str(declared).strip() == "":
            backend = infer_hf_backend(model_path)
        else:
            backend = str(declared).strip().lower() or "rnabert"
        if backend not in attention_backends:
            skipped_unsupported.add(name)
            continue
        if not name.startswith(("RIBOSPAN-", "AIDO.RNA-")):
            raise ValueError(
                f"Attention runner does not support registry model {name!r}"
            )
        entries.append(
            ModelEntry(
                name=name,
                path=model_path,
                context_window=raw.get("context_window", ""),
                registry=dict(raw),
                backend=backend,
            )
        )
    if requested:
        missing = sorted(
            requested
            - {entry.name for entry in entries}
            - skipped_unsupported
        )
        if missing:
            raise ValueError(f"Unknown --models registry names: {missing}")
    if not entries:
        if skipped_unsupported:
            return []
        raise ValueError("No models selected from registry")
    return entries


def load_manifest(path: Path, max_jobs: Optional[int]) -> List[Dict[str, Any]]:
    """Read and minimally validate a shared JSONL manifest."""

    if max_jobs is not None and max_jobs < 0:
        raise ValueError("--max-jobs must be non-negative")
    jobs: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Manifest row {line_number} is not an object")
            missing = [
                field
                for field in (
                    "job_id",
                    "pair_group_id",
                    "sequence",
                    "sequence_sha256",
                    "length",
                    "length_group",
                    "pattern_id",
                    "pattern_family",
                    "control_type",
                    "motif_start",
                    "motif_end",
                )
                if field not in value
            ]
            if missing:
                raise ValueError(
                    f"Manifest row {line_number} misses required fields: {missing}"
                )
            job_id = str(value["job_id"])
            if job_id in seen:
                raise ValueError(f"Duplicate manifest job_id: {job_id}")
            seen.add(job_id)
            value["job_id"] = job_id
            value["pair_group_id"] = str(value["pair_group_id"])
            value["length_group"] = int(value["length_group"])
            value["_clean_sequence"] = clean_sequence(value["sequence"])
            if not value["_clean_sequence"]:
                raise ValueError(f"Manifest job {job_id} has an empty sequence")
            if len(value["_clean_sequence"]) != int(value["length"]):
                raise ValueError(
                    f"Manifest job {job_id} length does not match its sequence"
                )
            digest = hashlib.sha256(str(value["sequence"]).encode("ascii")).hexdigest()
            if digest != str(value["sequence_sha256"]).lower():
                raise ValueError(
                    f"Manifest job {job_id} sequence_sha256 does not match"
                )
            jobs.append(value)
    return jobs


def _is_native(job: Mapping[str, Any]) -> bool:
    return str(job.get("control_type", "")).lower() == "native"


def expand_heatmap_jobs(
    requested_ids: Iterable[str],
    jobs: Sequence[Mapping[str, Any]],
) -> Set[str]:
    """Include both structured and native records from each requested pair group."""

    by_id = {str(job["job_id"]): job for job in jobs}
    requested = set(str(value) for value in requested_ids)
    missing = sorted(requested - set(by_id))
    if missing:
        raise ValueError(f"--heatmap-jobs not found in processed manifest: {missing}")
    unpaired = sorted(
        job_id
        for job_id in requested
        if not str(by_id[job_id].get("pair_group_id", ""))
    )
    if unpaired:
        raise ValueError(f"Heatmap jobs have empty pair_group_id: {unpaired}")
    groups = {
        str(by_id[job_id].get("pair_group_id", ""))
        for job_id in requested
        if str(by_id[job_id].get("pair_group_id", ""))
    }
    expanded = set(requested)
    for group in groups:
        members = [
            job for job in jobs if str(job.get("pair_group_id", "")) == group
        ]
        if not any(_is_native(job) for job in members) or not any(
            str(job.get("control_type", "")).lower() == "structured"
            for job in members
        ):
            raise ValueError(
                f"Heatmap pair_group_id={group!r} does not contain both structured "
                "and native jobs in the processed manifest"
            )
        expanded.update(
            str(job["job_id"])
            for job in members
            if str(job.get("control_type", "")).lower()
            in {"structured", "native"}
        )
    return expanded


def shared_heatmap_windows(
    jobs: Sequence[Mapping[str, Any]],
    selected_ids: Set[str],
    window_size: int,
) -> Dict[str, Tuple[int, int]]:
    """Use one interval for aligned structured/native heatmaps."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for job in jobs:
        if str(job["job_id"]) in selected_ids:
            grouped.setdefault(str(job.get("pair_group_id", "")), []).append(job)
    windows: Dict[str, Tuple[int, int]] = {}
    for group_jobs in grouped.values():
        reference: Optional[Tuple[int, int, int]] = None
        for job in group_jobs:
            sequence_length = len(str(job["_clean_sequence"]))
            motif = validate_motif_span(job, sequence_length)
            if not _is_native(job) and motif.valid:
                start, end = motif_centered_window(sequence_length, motif, window_size)
                reference = (start, end, sequence_length)
                break
        for job in group_jobs:
            job_id = str(job["job_id"])
            sequence_length = len(str(job["_clean_sequence"]))
            if reference is not None and reference[2] == sequence_length:
                windows[job_id] = (reference[0], reference[1])
            else:
                motif = validate_motif_span(job, sequence_length)
                windows[job_id] = motif_centered_window(
                    sequence_length, motif, window_size
                )
    return windows


def resolve_device(value: Optional[str]) -> torch.device:
    """Resolve and activate the requested inference device."""

    device = torch.device(value or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def resolve_dtype(value: str) -> torch.dtype:
    """Map CLI dtype names to torch dtypes."""

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _initial_metadata(
    args: argparse.Namespace,
    models: Sequence[ModelEntry],
    jobs: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "config": str(Path(args.config).resolve()),
        "models_registry": str(Path(args.registry).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "output": str(Path(args.output).resolve()),
        "selected_models": [entry.name for entry in models],
        "model_paths": {entry.name: str(entry.path) for entry in models},
        "job_count": len(jobs),
        "device": str(device),
        "dtype": args.dtype,
        "layers": _flatten_values(args.layers),
        "heads": _flatten_values(args.heads),
        "summary_only": bool(args.summary_only),
        "requested_heatmap_jobs": _flatten_values(args.heatmap_jobs),
        "query_chunk_size": args.query_chunk_size,
        "heatmap_window": args.heatmap_window,
        "heatmap_pool_size": args.heatmap_pool_size,
        "heatmap_norm_quantile": args.heatmap_norm_quantile,
        "resume_key": ["model_name", "job_id", "layer", "head"],
        "metric_definitions": METRIC_DEFINITIONS,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute diagnostics and return final run metadata."""

    project_root = Path(__file__).resolve().parents[1]
    if args.config is None:
        experiment_path = project_root / "configs" / "experiment.yaml"
    else:
        experiment_path = Path(args.config).expanduser()
        if not experiment_path.is_absolute():
            experiment_path = project_root / experiment_path
    experiment = load_experiment_config(experiment_path, profile=args.profile)
    attention_config = experiment.values["attention"]
    config_path = (
        Path(args.registry).expanduser().resolve()
        if args.registry
        else experiment.models_registry_path
    )
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else experiment.output_manifest_path
    )
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else project_root / "outputs" / experiment.profile / "attention"
    )
    args.config = str(experiment.config_path)
    args.registry = str(config_path)
    args.manifest = str(manifest_path)
    args.output = str(output_dir)
    args.dtype = args.dtype or str(attention_config.get("dtype", "bfloat16"))
    configured_layers = attention_config.get("layers", "all")
    configured_heads = attention_config.get("heads", "all")
    args.layers = args.layers or (
        [configured_layers]
        if isinstance(configured_layers, str)
        else [str(value) for value in configured_layers]
    )
    args.heads = args.heads or (
        [configured_heads]
        if isinstance(configured_heads, str)
        else [str(value) for value in configured_heads]
    )
    args.heatmap_window = args.heatmap_window or int(
        attention_config.get("heatmap_window", 512)
    )
    args.heatmap_pool_size = (
        args.heatmap_pool_size
        if args.heatmap_pool_size is not None
        else int(attention_config.get("heatmap_pool_size", 512))
    )
    args.heatmap_norm_quantile = (
        args.heatmap_norm_quantile
        if args.heatmap_norm_quantile is not None
        else float(attention_config.get("heatmap_norm_quantile", 0.995))
    )
    args.query_chunk_size = args.query_chunk_size or int(
        attention_config.get("query_chunk_size", 32)
    )
    models_dir = Path(args.models_dir).resolve() if args.models_dir else None
    if args.query_chunk_size < 1:
        raise ValueError("--query-chunk-size must be positive")
    if args.heatmap_window < 1:
        raise ValueError("--heatmap-window must be positive")
    if args.heatmap_pool_size < 0:
        raise ValueError("--heatmap-pool-size must be zero or positive")
    if not 0.0 < args.heatmap_norm_quantile <= 1.0:
        raise ValueError("--heatmap-norm-quantile must be in (0, 1]")

    model_entries = load_model_registry(config_path, args.models, models_dir)
    runnable_entries = []
    for entry in model_entries:
        available, reason = checkpoint_files_present(entry.path, entry.backend)
        if not available:
            print(f"skip {entry.name}: {reason}", flush=True)
            continue
        runnable_entries.append(entry)
    model_entries = runnable_entries
    if not model_entries:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no RNABert/RiboSpan models with checkpoints",
            "started_at": _utc_now(),
            "finished_at": _utc_now(),
            "config": str(experiment.config_path),
            "models_registry": str(config_path),
            "manifest": str(manifest_path),
            "output": str(output_dir),
            "selected_models": [],
        }
        _write_json(output_dir / "run_metadata.json", metadata)
        print(
            "Skipping attention: no RNABert/RiboSpan models with checkpoints",
            flush=True,
        )
        return metadata
    jobs = load_manifest(manifest_path, args.max_jobs)
    if not jobs:
        raise ValueError("No manifest jobs remain after filtering")
    requested_real_tokens = max(len(str(job["_clean_sequence"])) for job in jobs)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    summary_path = output_dir / "attention_summary.csv"
    metadata_path = output_dir / "run_metadata.json"
    heatmap_metadata_path = output_dir / "heatmaps" / "heatmap_metadata.json"

    requested_heatmaps = set()
    if not args.summary_only:
        requested_heatmaps = expand_heatmap_jobs(
            _flatten_values(args.heatmap_jobs), jobs
        )
    windows = shared_heatmap_windows(
        jobs, requested_heatmaps, args.heatmap_window
    )
    metadata = _initial_metadata(args, model_entries, jobs, device)
    metadata["expanded_heatmap_jobs"] = sorted(requested_heatmaps)
    fingerprint_payload = {
        "config_sha256": _file_digest(Path(args.config)),
        "registry_sha256": _file_digest(Path(args.registry)),
        "manifest_sha256": _file_digest(Path(args.manifest)),
        "dtype": args.dtype,
        "query_chunk_size": args.query_chunk_size,
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    metadata["run_fingerprint"] = run_fingerprint
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / ".attention.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            f"Another attention run is already writing to {output_dir}"
        ) from exc
    if summary_path.exists() and summary_path.stat().st_size > 0:
        if not metadata_path.exists():
            raise ValueError(
                "Existing attention summary has no run metadata; use a new "
                "--output directory instead of resuming it."
            )
        try:
            previous_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Cannot validate attention resume metadata: {metadata_path}"
            ) from exc
        previous_fingerprint = previous_metadata.get("run_fingerprint")
        if str(previous_fingerprint or "") != run_fingerprint:
            raise ValueError(
                "Refusing to resume attention output with a different run "
                f"fingerprint: existing={previous_fingerprint}, "
                f"requested={run_fingerprint}. Use a new --output directory."
            )
    _write_json(metadata_path, metadata)

    completed = load_completed_keys(summary_path)
    initially_completed = len(completed)
    heatmap_metadata: List[Dict[str, Any]] = []
    if heatmap_metadata_path.exists():
        try:
            previous_heatmaps = json.loads(
                heatmap_metadata_path.read_text(encoding="utf-8")
            )
            heatmap_metadata = [
                dict(item)
                for item in previous_heatmaps.get("captures", [])
                if isinstance(item, Mapping)
            ]
        except (json.JSONDecodeError, OSError, TypeError):
            heatmap_metadata = []
    initial_heatmap_count = len(heatmap_metadata)
    plotting_available = True
    processed_forwards = 0
    skipped_complete_jobs = 0
    started = time.time()

    try:
        tokenizer_by_backend: Dict[str, Any] = {}
        with StreamingCSVWriter(summary_path, SUMMARY_FIELDS) as writer:
            collector = AttentionSummaryCollector(
                row_callback=writer.write,
                query_chunk_size=args.query_chunk_size,
                completed_keys=completed,
            )
            for model_index, entry in enumerate(model_entries, start=1):
                print(
                    f"[{model_index}/{len(model_entries)}] loading {entry.name} "
                    f"from {entry.path} on {device} "
                    f"(FP32 weights, {args.dtype} autocast, backend={entry.backend})",
                    flush=True,
                )
                rope_scaling = normalize_rope_scaling_policy(
                    entry.registry.get("rope_scaling")
                )
                if entry.backend not in tokenizer_by_backend:
                    tokenizer_by_backend[entry.backend] = load_rnabert_tokenizer(
                        entry.backend
                    )
                tokenizer = tokenizer_by_backend[entry.backend]
                model = load_rnabert_model(
                    entry.path,
                    dtype=dtype,
                    device=device,
                    requested_real_tokens=requested_real_tokens,
                    backend=entry.backend,
                )
                metadata.setdefault("model_runtime", {})[entry.name] = {
                    "backend": entry.backend,
                    "trained_context_window": entry.context_window,
                    "checkpoint_config_max_position_embeddings": getattr(
                        model,
                        "_benchmark_original_max_position_embeddings",
                    ),
                    "runtime_max_position_embeddings": getattr(
                        model,
                        "_benchmark_runtime_max_position_embeddings",
                    ),
                    "requested_real_token_length": requested_real_tokens,
                    "position_embedding_type": model.config.position_embedding_type,
                    "weight_dtype": str(next(model.parameters()).dtype),
                    "autocast_dtype": str(dtype),
                    "rope_scaling_policy": rope_scaling,
                }
                if rope_scaling is not None:
                    factors = [
                        runtime_rope_scaling_factor(
                            len(str(job["_clean_sequence"])), rope_scaling
                        )
                        for job in jobs
                    ]
                    metadata["model_runtime"][entry.name].update(
                        yarn_factor_min=min(factors),
                        yarn_factor_max=max(factors),
                    )
                _write_json(metadata_path, metadata)
                num_layers = int(model.config.num_hidden_layers)
                num_heads = int(model.config.num_attention_heads)
                selected_layers = _parse_indices(args.layers, num_layers, "layer")
                summary_heads, heatmap_heads, heatmap_all = _head_selections(
                    args.heads, num_heads
                )
                if requested_heatmaps and (
                    "all"
                    in {value.lower() for value in _flatten_values(args.layers)}
                    or heatmap_all
                ):
                    raise ValueError(
                        "Heatmaps require explicit --layers and --heads "
                        "(use --heads mean for a mean-head heatmap)."
                    )

                with torch.inference_mode():
                    for job_index, job in enumerate(jobs, start=1):
                        job_id = str(job["job_id"])
                        wants_heatmap = job_id in requested_heatmaps
                        heatmap_layers = selected_layers if wants_heatmap else set()
                        active_layers = {
                            layer
                            for layer in selected_layers
                            if wants_heatmap
                            or any(
                                (entry.name, job_id, layer, head) not in completed
                                for head in summary_heads
                            )
                        }
                        if not active_layers:
                            skipped_complete_jobs += 1
                            continue
                        sequence = str(job["_clean_sequence"])
                        configure_runtime_rope_scaling(
                            model,
                            real_sequence_length=len(sequence),
                            policy=rope_scaling,
                        )
                        collector.begin_job(
                            model_name=entry.name,
                            model_path=entry.path,
                            model_context_window=entry.context_window,
                            job=job,
                            sequence_length=len(sequence),
                            summary_heads=summary_heads,
                            heatmap_layers=heatmap_layers,
                            heatmap_heads=heatmap_heads if wants_heatmap else (),
                            heatmap_window=windows.get(job_id),
                            heatmap_window_size=args.heatmap_window,
                        )
                        batch = encode_sequence(tokenizer, sequence, device)
                        with patched_rnabert_attention(
                            model, collector, active_layers
                        ):
                            # Skip the MLM head (would allocate seq x hidden).
                            autocast_context = (
                                torch.autocast(
                                    device_type=device.type,
                                    dtype=dtype,
                                )
                                if dtype != torch.float32
                                else nullcontext()
                            )
                            with autocast_context:
                                outputs = hf_encoder(model)(
                                    **batch,
                                    output_attentions=False,
                                    return_dict=True,
                                )
                        del outputs, batch
                        processed_forwards += 1

                        for capture in collector.pop_heatmaps():
                            if plotting_available:
                                try:
                                    heatmap_metadata.append(
                                        save_heatmap(
                                            capture,
                                            output_dir / "heatmaps",
                                            pool_size=args.heatmap_pool_size,
                                            norm_quantile=args.heatmap_norm_quantile,
                                        )
                                    )
                                except (ImportError, ModuleNotFoundError) as exc:
                                    plotting_available = False
                                    metadata["heatmap_warning"] = (
                                        "Plotting dependency unavailable; summaries "
                                        f"completed without heatmaps: {exc}"
                                    )
                                    print(metadata["heatmap_warning"], flush=True)
                        print(
                            f"model={entry.name} job={job_index}/{len(jobs)} "
                            f"id={job_id} length={len(sequence)} "
                            f"layers={len(active_layers)}",
                            flush=True,
                        )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if heatmap_metadata:
            deduplicated: Dict[tuple[Any, ...], Dict[str, Any]] = {}
            for item in heatmap_metadata:
                key = (
                    item.get("model_name"),
                    item.get("job_id"),
                    item.get("layer"),
                    item.get("head"),
                    item.get("window_start"),
                    item.get("window_end"),
                )
                deduplicated[key] = item
            heatmap_metadata = list(deduplicated.values())
            _write_json(
                heatmap_metadata_path,
                {
                    "schema_version": 1,
                    "normalization": METRIC_DEFINITIONS["row_normalization"],
                    "captures": heatmap_metadata,
                },
            )
        metadata.update(
            {
                "status": "complete",
                "finished_at": _utc_now(),
                "elapsed_seconds": time.time() - started,
                "resume_rows_before_run": initially_completed,
                "summary_rows_after_run": len(completed),
                "new_summary_rows": len(completed) - initially_completed,
                "processed_forwards": processed_forwards,
                "skipped_complete_jobs": skipped_complete_jobs,
                "heatmap_count": len(heatmap_metadata) - initial_heatmap_count,
                "heatmap_count_total": len(heatmap_metadata),
                "plotting_available": plotting_available,
            }
        )
        _write_json(metadata_path, metadata)
        lock_handle.close()
        return metadata
    except BaseException as exc:
        # Rows are flushed individually, so the CSV stays resumable.
        metadata.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "elapsed_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resume_rows_before_run": initially_completed,
                "summary_rows_after_run": len(completed),
                "new_summary_rows": len(completed) - initially_completed,
                "processed_forwards": processed_forwards,
                "skipped_complete_jobs": skipped_complete_jobs,
                "heatmap_count": len(heatmap_metadata),
            }
        )
        _write_json(metadata_path, metadata)
        lock_handle.close()
        raise


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    metadata = run(args)
    print(
        f"Wrote {metadata['new_summary_rows']} new summary rows to "
        f"{Path(args.output).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
