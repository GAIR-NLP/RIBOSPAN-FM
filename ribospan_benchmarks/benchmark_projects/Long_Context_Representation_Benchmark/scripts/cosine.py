# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Layer-wise RNABert representation cosine runner."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import load_experiment_config, public_path

from .metrics import (
    context_cosine_rows,
    diffusion_cosine_rows,
    normalize_distance_bins,
    sample_token_trajectories,
    summarize_context_rows,
)
from .model_io import (
    LoadedRNABert,
    ModelSpec,
    backend_runtime_available,
    dtype_from_name,
    load_encoder,
    load_model_registry,
    normalize_sequence,
    resolve_device,
    runtime_rope_scaling_factor,
    weights_available,
)


REQUIRED_MANIFEST_FIELDS = {
    "job_id",
    "pair_group_id",
    "seed",
    "length",
    "length_group",
    "source_type",
    "background_type",
    "pattern_id",
    "pattern_family",
    "strength_mode",
    "strength_value",
    "motif_start",
    "motif_end",
    "control_type",
    "sequence",
    "sequence_sha256",
    "degenerate_control",
}
REQUIRED_CONTROL_TYPES = {"structured", "native"}
ALLOWED_CONTROL_TYPES = set(REQUIRED_CONTROL_TYPES)

CONTEXT_FIELDS = [
    "model",
    "model_path",
    "job_id",
    "pair_group_id",
    "seed",
    "length",
    "length_group",
    "source_type",
    "background_type",
    "strength_mode",
    "strength_value",
    "motif_start",
    "motif_end",
    "pattern_id",
    "pattern_family",
    "control_type",
    "degenerate_control",
    "layer",
    "layer_name",
    "base",
    "metric_variant",
    "cross_context_same_base_cos",
    "within_motif_same_base_cos",
    "within_background_same_base_cos",
    "same_context_baseline_cos",
    "context_separation",
    "anisotropy_baseline",
    "anisotropy_baseline_n_pairs",
    "n_cross_pairs",
    "n_within_motif_pairs",
    "n_within_background_pairs",
    "n_motif_positions",
    "n_background_positions",
]

TRAJECTORY_FIELDS = [
    "model",
    "model_path",
    "job_id",
    "pair_group_id",
    "seed",
    "length",
    "length_group",
    "source_type",
    "background_type",
    "strength_mode",
    "strength_value",
    "motif_start",
    "motif_end",
    "pattern_id",
    "pattern_family",
    "control_type",
    "degenerate_control",
    "layer",
    "layer_name",
    "position",
    "position_rank",
    "region",
    "base",
    "distance_to_motif",
    "l2_norm",
    "cos_to_embedding",
    "step_cosine_distance",
    "centroid_margin",
    "projection_1",
    "projection_2",
    "projection_3",
]

DIFFUSION_FIELDS = [
    "model",
    "model_path",
    "pair_group_id",
    "structured_job_id",
    "control_job_id",
    "control_type",
    "pattern_id",
    "pattern_family",
    "seed",
    "length",
    "length_group",
    "source_type",
    "background_type",
    "strength_mode",
    "strength_value",
    "motif_start",
    "motif_end",
    "degenerate_control",
    "layer",
    "layer_name",
    "metric_variant",
    "bin_index",
    "distance_bin",
    "distance_min",
    "distance_max",
    "mean_distance",
    "mean",
    "std",
    "n_positions",
    "n_positions_total",
    "local_peak",
    "far_field",
    "relative_distal",
    "relative_distal_threshold",
    "relative_distal_max_distance",
    "relative_distal_n_positions",
    "relative_distal_n_positions_total",
    "relative_distal_025",
    "relative_distal_025_n_positions",
    "relative_distal_025_n_positions_total",
    "relative_distal_050",
    "relative_distal_050_n_positions",
    "relative_distal_050_n_positions_total",
    "relative_distal_075",
    "relative_distal_075_n_positions",
    "relative_distal_075_n_positions_total",
    "leakage_auc",
    "anisotropy_baseline_motif",
    "anisotropy_baseline_control",
    "anisotropy_pair_baseline",
    "n_changed_motif_excluded",
    "n_changed_total_excluded",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse experiment arguments; paths never default relative to the CWD."""

    parser = argparse.ArgumentParser(
        description="Run layer-wise contextualization and causal diffusion probes."
    )
    parser.add_argument("--config", required=True, help="Run YAML configuration.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Experiment profile; defaults to default_profile in experiment.yaml.",
    )
    parser.add_argument(
        "--models",
        dest="models_path",
        default=None,
        help="Override the models.yaml registry path.",
    )
    parser.add_argument("--manifest", default=None, help="Override JSONL manifest.")
    parser.add_argument("--output", default=None, help="Override output directory.")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Run only this registry model; repeat to select multiple models.",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=None,
        help="Run only this job_id; repeat to select multiple jobs.",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        help="Additional job_id filter list.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default=None,
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume without duplicating completed output keys (default: true).",
    )
    return parser.parse_args(argv)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Relative configs are project-root relative, never process-CWD relative.
    return (project_root / path).resolve()


def _resolve_input_path(
    raw_path: str | Path,
    *,
    project_root: Path,
    config_dir: Path | None = None,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    parents = (project_root,) if config_dir is None else (project_root, config_dir)
    for parent in parents:
        candidate = (parent / path).resolve()
        if candidate.exists():
            return candidate
    return (project_root / path).resolve()


def _resolve_output_path(
    raw_path: str | Path,
    *,
    project_root: Path,
) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{field} must be a boolean, got {value!r}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load, normalize, and strictly validate all JSONL pair groups."""

    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Manifest line {line_number} must be an object")
            missing = sorted(REQUIRED_MANIFEST_FIELDS - raw.keys())
            if missing:
                raise ValueError(
                    f"Manifest line {line_number} missing fields: {missing}"
                )

            job = dict(raw)
            for key in (
                "job_id",
                "pair_group_id",
                "pattern_id",
                "pattern_family",
                "control_type",
            ):
                job[key] = str(job[key])
            if job["job_id"] in seen_ids:
                raise ValueError(f"Duplicate job_id: {job['job_id']}")
            seen_ids.add(job["job_id"])
            if job["control_type"] not in ALLOWED_CONTROL_TYPES:
                raise ValueError(
                    f"job {job['job_id']}: invalid control_type "
                    f"{job['control_type']!r}"
                )

            for key in (
                "seed",
                "length",
                "length_group",
                "motif_start",
                "motif_end",
            ):
                job[key] = int(job[key])
            job["degenerate_control"] = _as_bool(
                job["degenerate_control"], "degenerate_control"
            )
            if not isinstance(job["sequence"], str):
                raise ValueError(f"job {job['job_id']}: sequence must be a string")
            # Validate model-token compatibility, but retain exact manifest text
            # for its SHA-256 integrity check.
            normalized_sequence = normalize_sequence(job["sequence"])
            if len(normalized_sequence) != job["length"]:
                raise ValueError(
                    f"job {job['job_id']}: declared length={job['length']} but "
                    f"sequence length={len(normalized_sequence)}"
                )
            if not (
                0
                <= job["motif_start"]
                < job["motif_end"]
                <= job["length"]
            ):
                raise ValueError(
                    f"job {job['job_id']}: invalid motif interval "
                    f"[{job['motif_start']}, {job['motif_end']})"
                )
            digest = hashlib.sha256(job["sequence"].encode("utf-8")).hexdigest()
            if digest.lower() != str(job["sequence_sha256"]).lower():
                raise ValueError(
                    f"job {job['job_id']}: sequence_sha256 does not match sequence"
                )
            job["sequence"] = normalized_sequence
            job["sequence_sha256"] = digest
            jobs.append(job)
    if not jobs:
        raise ValueError("Manifest is empty")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[job["pair_group_id"]].append(job)
    for group_id, group in grouped.items():
        counts: dict[str, int] = defaultdict(int)
        for job in group:
            counts[job["control_type"]] += 1
        bad = {
            control_type: counts.get(control_type, 0)
            for control_type in sorted(REQUIRED_CONTROL_TYPES)
            if counts.get(control_type, 0) != 1
        }
        if bad:
            raise ValueError(
                f"pair_group_id={group_id!r} must have exactly one of each "
                f"{sorted(REQUIRED_CONTROL_TYPES)}; bad counts={bad}"
            )
        duplicates = {
            control_type: count
            for control_type, count in counts.items()
            if count > 1
        }
        if duplicates:
            raise ValueError(
                f"pair_group_id={group_id!r} has duplicate controls: "
                f"{duplicates}"
            )
        lengths = {job["length"] for job in group}
        intervals = {
            (job["motif_start"], job["motif_end"]) for job in group
        }
        if len(lengths) != 1 or len(intervals) != 1:
            raise ValueError(
                f"pair_group_id={group_id!r} has unaligned lengths/motif intervals"
            )
    return jobs


def _stable_seed(*parts: Any) -> int:
    text = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _job_metric_seed(job: Mapping[str, Any], metric: str) -> int:
    """Return a model-independent seed for one manifest job."""

    return _stable_seed(job["job_id"], job["seed"], metric)


def _pair_metric_seed(
    left_job: Mapping[str, Any],
    right_job: Mapping[str, Any],
    metric: str,
) -> int:
    """Return a model-independent seed for one matched intervention pair."""

    return _stable_seed(left_job["job_id"], right_job["job_id"], metric)


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class CsvSink:
    """Append-only CSV sink with resume-time completed-key de-duplication."""

    def __init__(
        self,
        path: Path,
        fieldnames: Sequence[str],
        key_fields: Sequence[str],
        *,
        resume: bool,
    ) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.key_fields = tuple(key_fields)
        self.completed: set[tuple[str, ...]] = set()
        path.parent.mkdir(parents=True, exist_ok=True)

        if resume and path.exists() and path.stat().st_size > 0:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != self.fieldnames:
                    raise ValueError(
                        f"Existing CSV schema does not match this implementation: {path}"
                    )
                for row in reader:
                    self.completed.add(self._key(row))
            self.handle = path.open("a", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(
                self.handle, fieldnames=self.fieldnames, extrasaction="ignore"
            )
        else:
            self.handle = path.open("w", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(
                self.handle, fieldnames=self.fieldnames, extrasaction="ignore"
            )
            self.writer.writeheader()
            self.handle.flush()

    def _key(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(field, "")) for field in self.key_fields)

    def append(self, rows: Iterable[Mapping[str, Any]]) -> int:
        written = 0
        for row in rows:
            key = self._key(row)
            if key in self.completed:
                continue
            serialized = {
                field: _csv_value(row.get(field)) for field in self.fieldnames
            }
            self.writer.writerow(serialized)
            self.completed.add(key)
            written += 1
        self.handle.flush()
        return written

    def close(self) -> None:
        self.handle.close()


class JsonlSink:
    """JSONL sink keyed by an explicit completion_key."""

    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.completed: set[str] = set()
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid resume JSONL at {path}:{line_number}"
                        ) from exc
                    key = value.get("completion_key")
                    if key is not None:
                        self.completed.add(str(key))
            self.handle = path.open("a", encoding="utf-8")
        else:
            self.handle = path.open("w", encoding="utf-8")

    def append(self, value: Mapping[str, Any]) -> bool:
        key = str(value["completion_key"])
        if key in self.completed:
            return False
        self.handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self.handle.flush()
        self.completed.add(key)
        return True

    def close(self) -> None:
        self.handle.close()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _job_prefix(
    model_spec: ModelSpec,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_spec.name,
        "model_path": str(model_spec.path),
        "job_id": job["job_id"],
        "pair_group_id": job["pair_group_id"],
        "seed": job["seed"],
        "length": job["length"],
        "length_group": job["length_group"],
        "source_type": job.get("source_type"),
        "background_type": job.get("background_type"),
        "pattern_id": job["pattern_id"],
        "pattern_family": job["pattern_family"],
        "strength_mode": job.get("strength_mode"),
        "strength_value": job.get("strength_value"),
        "motif_start": job["motif_start"],
        "motif_end": job["motif_end"],
        "control_type": job["control_type"],
        "degenerate_control": job["degenerate_control"],
    }


def _diffusion_prefix(
    model_spec: ModelSpec,
    motif_job: Mapping[str, Any],
    control_job: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_spec.name,
        "model_path": str(model_spec.path),
        "pair_group_id": motif_job["pair_group_id"],
        "structured_job_id": motif_job["job_id"],
        "control_job_id": control_job["job_id"],
        "control_type": control_job["control_type"],
        "pattern_id": motif_job["pattern_id"],
        "pattern_family": motif_job["pattern_family"],
        "seed": motif_job["seed"],
        "length": motif_job["length"],
        "length_group": motif_job["length_group"],
        "source_type": motif_job.get("source_type"),
        "background_type": motif_job.get("background_type"),
        "strength_mode": motif_job.get("strength_mode"),
        "strength_value": motif_job.get("strength_value"),
        "motif_start": motif_job["motif_start"],
        "motif_end": motif_job["motif_end"],
        "degenerate_control": control_job["degenerate_control"],
    }


def _add_prefix(
    rows: Sequence[Mapping[str, Any]],
    prefix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [{**prefix, **row} for row in rows]


def _diffusion_layer_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (int(row["layer"]), str(row["metric_variant"]))
        if key in seen:
            continue
        seen.add(key)
        summaries.append(
            {
                "layer": key[0],
                "layer_name": row["layer_name"],
                "metric_variant": key[1],
                "local_peak": row["local_peak"],
                "far_field": row["far_field"],
                "relative_distal": row["relative_distal"],
                "relative_distal_025": row["relative_distal_025"],
                "relative_distal_050": row["relative_distal_050"],
                "relative_distal_075": row["relative_distal_075"],
                "leakage_auc": row["leakage_auc"],
            }
        )
    return summaries


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _merge_names(previous: Sequence[Any], current: Sequence[str]) -> list[str]:
    names = [str(name) for name in previous if name]
    names.extend(str(name) for name in current)
    return list(dict.fromkeys(names))


def _merge_skipped(
    previous: Sequence[Any],
    current: Sequence[Mapping[str, str]],
    completed: set[str],
) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for item in previous:
        if isinstance(item, Mapping) and item.get("model"):
            name = str(item["model"])
            if name not in completed:
                merged[name] = dict(item)
        elif item and str(item) not in completed:
            merged[str(item)] = {"model": str(item)}
    for item in current:
        name = str(item["model"])
        if name not in completed:
            merged[name] = dict(item)
    return list(merged.values())


def _analyze_job(
    loaded: LoadedRNABert,
    model_spec: ModelSpec,
    job: Mapping[str, Any],
    context_config: Mapping[str, Any],
    context_sink: CsvSink,
    trajectory_sink: CsvSink | None,
    summary_sink: JsonlSink,
    fingerprint: str,
) -> tuple[tuple[Any, ...], list[float], list[dict[str, Any]]]:
    hidden_states = loaded.hidden_states(str(job["sequence"]))
    metric_rows, baselines = context_cosine_rows(
        hidden_states,
        str(job["sequence"]),
        int(job["motif_start"]),
        int(job["motif_end"]),
        max_positions_per_context=int(
            context_config.get("max_positions_per_context", 512)
        ),
        max_pairs_per_metric=int(
            context_config.get("max_pairs_per_metric", 4096)
        ),
        anisotropy_pairs=int(context_config.get("anisotropy_pairs", 4096)),
        anisotropy_match_base=_as_bool(
            context_config.get("anisotropy_match_base", True),
            "context.anisotropy_match_base",
        ),
        motif_interior_trim=int(context_config.get("motif_interior_trim", 0)),
        background_exclusion_radius=int(
            context_config.get("background_exclusion_radius", 0)
        ),
        seed=_job_metric_seed(job, "context"),
    )
    prefixed = _add_prefix(metric_rows, _job_prefix(model_spec, job))
    rows_written = context_sink.append(prefixed)
    trajectory_rows_written = 0
    trajectory_config = context_config.get("trajectory", {})
    if trajectory_sink is not None and isinstance(trajectory_config, Mapping):
        trajectory_rows = sample_token_trajectories(
            hidden_states,
            str(job["sequence"]),
            int(job["motif_start"]),
            int(job["motif_end"]),
            positions_per_base_region=int(
                trajectory_config.get("positions_per_base_region", 2)
            ),
            background_exclusion_radius=int(
                context_config.get("background_exclusion_radius", 0)
            ),
            projection_dim=int(trajectory_config.get("projection_dim", 3)),
            projection_seed=int(
                trajectory_config.get("projection_seed", 1729)
            ),
            seed=_job_metric_seed(job, "trajectory"),
        )
        trajectory_rows_written = trajectory_sink.append(
            _add_prefix(trajectory_rows, _job_prefix(model_spec, job))
        )
    summary_sink.append(
        {
            "completion_key": (
                f"job|{fingerprint}|{model_spec.name}|{job['job_id']}"
            ),
            "record_type": "job",
            **_job_prefix(model_spec, job),
            "n_layers_including_embedding": len(hidden_states),
            "context_rows_written_this_run": rows_written,
            "trajectory_rows_written_this_run": trajectory_rows_written,
            "context_layer_summary": summarize_context_rows(metric_rows),
            "statistical_unit": "job",
        }
    )
    return hidden_states, baselines, metric_rows


def _run_group(
    loaded: LoadedRNABert,
    model_spec: ModelSpec,
    jobs: Sequence[Mapping[str, Any]],
    context_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    context_sink: CsvSink,
    diffusion_sink: CsvSink,
    trajectory_sink: CsvSink | None,
    summary_sink: JsonlSink,
    fingerprint: str,
) -> None:
    motif_job = next(
        (job for job in jobs if job["control_type"] == "structured"), None
    )
    if motif_job is None:
        for job in jobs:
            hidden, _, _ = _analyze_job(
                loaded,
                model_spec,
                job,
                context_config,
                context_sink,
                trajectory_sink,
                summary_sink,
                fingerprint,
            )
            del hidden
            gc.collect()
        return

    motif_hidden, motif_baselines, _ = _analyze_job(
        loaded,
        model_spec,
        motif_job,
        context_config,
        context_sink,
        trajectory_sink,
        summary_sink,
        fingerprint,
    )
    controls = [job for job in jobs if job["control_type"] != "structured"]
    for control_job in controls:
        control_hidden, control_baselines, _ = _analyze_job(
            loaded,
            model_spec,
            control_job,
            context_config,
            context_sink,
            trajectory_sink,
            summary_sink,
            fingerprint,
        )
        diffusion_rows = diffusion_cosine_rows(
            motif_hidden,
            control_hidden,
            str(motif_job["sequence"]),
            str(control_job["sequence"]),
            int(motif_job["motif_start"]),
            int(motif_job["motif_end"]),
            distance_bins=diffusion_config.get(
                "distance_bins",
                diffusion_config.get(
                    "bins", [0, 8, 32, 128, 512, 1024, 2048, 4096, "inf"]
                ),
            ),
            far_field_threshold=float(
                diffusion_config.get("far_field_threshold", 1024)
            ),
            relative_distal_threshold=float(
                diffusion_config.get("relative_distal_threshold", 0.75)
            ),
            relative_distal_thresholds=diffusion_config.get(
                "relative_distal_thresholds", [0.25, 0.50, 0.75]
            ),
            max_positions_per_bin=int(
                diffusion_config.get("max_positions_per_bin", 4096)
            ),
            motif_anisotropy_baselines=motif_baselines,
            control_anisotropy_baselines=control_baselines,
            seed=_pair_metric_seed(
                motif_job, control_job, "diffusion"
            ),
        )
        prefixed = _add_prefix(
            diffusion_rows,
            _diffusion_prefix(model_spec, motif_job, control_job),
        )
        rows_written = diffusion_sink.append(prefixed)
        summary_sink.append(
            {
                "completion_key": (
                    f"diffusion|{fingerprint}|{model_spec.name}|"
                    f"{motif_job['job_id']}|{control_job['job_id']}"
                ),
                "record_type": "diffusion_pair",
                **_diffusion_prefix(model_spec, motif_job, control_job),
                "diffusion_rows_written_this_run": rows_written,
                "diffusion_layer_summary": _diffusion_layer_summary(
                    diffusion_rows
                ),
                "statistical_unit": "matched_pair_group_comparison",
            }
        )
        del control_hidden, diffusion_rows, prefixed
        gc.collect()
    del motif_hidden
    gc.collect()


def run(args: argparse.Namespace) -> None:
    """Execute selected jobs, loading and releasing exactly one model at a time."""

    project_root = _project_root()
    config_path = _config_path(args.config, project_root)
    experiment = load_experiment_config(config_path, profile=args.profile)
    config = dict(experiment.values)
    config_dir = experiment.config_path.parent

    raw_models_path = args.models_path or experiment.models_registry_path
    raw_manifest_path = args.manifest or experiment.output_manifest_path
    raw_output_path = args.output or (
        project_root / "outputs" / experiment.profile / "cosine"
    )

    models_path = _resolve_input_path(
        raw_models_path, project_root=project_root, config_dir=config_dir
    )
    manifest_path = _resolve_input_path(
        raw_manifest_path, project_root=project_root, config_dir=config_dir
    )
    output_dir = _resolve_output_path(raw_output_path, project_root=project_root)
    if not models_path.is_file():
        raise FileNotFoundError(f"models.yaml not found: {models_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_model_registry(
        models_path, config_dir=config_dir, project_root=project_root
    )
    execution_config = config.get("execution", {})
    selected_model_names = args.model or (
        execution_config.get("models")
        if isinstance(execution_config, Mapping)
        else None
    )
    if selected_model_names:
        if isinstance(selected_model_names, str):
            selected_model_names = [selected_model_names]
        selected_set = {str(name) for name in selected_model_names}
        unknown = sorted(selected_set - registry.keys())
        if unknown:
            raise ValueError(f"Unknown model filters: {unknown}")
        registry = {
            name: spec for name, spec in registry.items() if name in selected_set
        }

    requested_model_names = list(registry)
    skipped_models: list[dict[str, str]] = []
    runnable: dict[str, Any] = {}
    for name, spec in registry.items():
        available, reason = backend_runtime_available(spec.backend)
        if available:
            available, reason = weights_available(spec)
        if not available:
            record = {
                "model": name,
                "backend": spec.backend or "rnabert",
                "reason": reason,
            }
            skipped_models.append(record)
            print(f"skip {name} ({spec.backend}): {reason}", flush=True)
            continue
        runnable[name] = spec
    registry = runnable

    manifest_jobs = load_manifest(manifest_path)
    requested_jobs = list(args.job or []) + list(args.jobs or [])
    if not requested_jobs:
        configured_jobs = (
            execution_config.get("jobs")
            if isinstance(execution_config, Mapping)
            else None
        )
        if configured_jobs:
            if isinstance(configured_jobs, str):
                requested_jobs = [configured_jobs]
            else:
                requested_jobs = [str(job_id) for job_id in configured_jobs]
    if requested_jobs:
        requested_set = set(requested_jobs)
        known_jobs = {job["job_id"] for job in manifest_jobs}
        unknown_jobs = sorted(requested_set - known_jobs)
        if unknown_jobs:
            raise ValueError(f"Unknown job filters: {unknown_jobs}")
        manifest_jobs = [
            job for job in manifest_jobs if job["job_id"] in requested_set
        ]
    max_jobs = (
        args.max_jobs
        if args.max_jobs is not None
        else (
            execution_config.get("max_jobs")
            if isinstance(execution_config, Mapping)
            else None
        )
    )
    if max_jobs is not None:
        max_jobs = int(max_jobs)
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        manifest_jobs = manifest_jobs[:max_jobs]
    if not manifest_jobs:
        raise ValueError("No jobs remain after filtering")

    cosine_config = config.get("cosine", {})
    if not isinstance(cosine_config, Mapping):
        raise ValueError("cosine configuration must be a mapping")
    dtype_name = args.dtype or str(cosine_config.get("dtype", "bfloat16"))
    device_name = args.device or str(cosine_config.get("device", "auto"))
    dtype = dtype_from_name(dtype_name)
    device = resolve_device(device_name)
    resume = (
        args.resume
        if args.resume is not None
        else _as_bool(cosine_config.get("resume", True), "cosine.resume")
    )
    requested_max_length = max(int(job["length"]) for job in manifest_jobs)

    context_config = dict(cosine_config.get("context", {}))
    diffusion_config = dict(cosine_config.get("diffusion", {}))
    sampling_config = config["sampling"]
    context_config["anisotropy_pairs"] = int(
        sampling_config["anisotropy_pair_sample_size"]
    )
    context_config["max_pairs_per_metric"] = int(
        sampling_config["context_pair_sample_size"]
    )
    diffusion_config["distance_bins"] = sampling_config[
        "representation_distance_bins"
    ]
    diffusion_config["far_field_threshold"] = sampling_config[
        "far_field_threshold"
    ]
    diffusion_config["max_positions_per_bin"] = sampling_config[
        "max_positions_per_bin"
    ]
    normalize_distance_bins(
        diffusion_config.get(
            "distance_bins",
            diffusion_config.get(
                "bins", [0, 8, 32, 128, 512, 1024, 2048, 4096, "inf"]
            ),
        )
    )
    far_threshold = float(diffusion_config.get("far_field_threshold", 1024))
    if far_threshold < 0:
        raise ValueError("diffusion.far_field_threshold must be non-negative")

    fingerprint_payload = {
        "config_sha256": _file_digest(config_path),
        "models_sha256": _file_digest(models_path),
        "manifest_sha256": _file_digest(manifest_path),
        "dtype": dtype_name,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    metadata_path = output_dir / "run_metadata.json"
    previous_metadata: dict[str, Any] = {}
    if resume and metadata_path.exists():
        try:
            previous_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Cannot validate resume metadata: {metadata_path}"
            ) from exc
        previous_fingerprint = previous_metadata.get("run_fingerprint")
        if (
            previous_fingerprint is not None
            and str(previous_fingerprint) != fingerprint
        ):
            raise ValueError(
                "Refusing to resume cosine output with a different run "
                f"fingerprint: existing={previous_fingerprint}, "
                f"requested={fingerprint}. Use a new --output directory."
            )

    previous_selected = list(previous_metadata.get("selected_models") or [])
    previous_runtime = dict(previous_metadata.get("model_runtime") or {})
    previous_skipped = list(previous_metadata.get("skipped_models") or [])
    merged_selected = _merge_names(previous_selected, requested_model_names)
    merged_skipped = _merge_skipped(
        previous_skipped, skipped_models, set(registry)
    )
    if not registry:
        started_at = datetime.now(timezone.utc).isoformat()
        skip_only = {
            "status": "completed",
            "started_at": previous_metadata.get("started_at", started_at),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "run_fingerprint": fingerprint,
            "config_path": public_path(config_path, project_root),
            "models_path": public_path(models_path, project_root),
            "manifest_path": public_path(manifest_path, project_root),
            "output_dir": public_path(output_dir, project_root),
            "selected_models": merged_selected,
            "skipped_models": merged_skipped,
            "model_runtime": previous_runtime,
            "selected_jobs": [job["job_id"] for job in manifest_jobs],
            "device": str(device),
            "dtype": dtype_name,
            "resume": resume,
            "note": "all selected models skipped in this environment",
        }
        _write_metadata(metadata_path, skip_only)
        print(
            "no runnable models in this environment; recorded skips in "
            f"{metadata_path}",
            flush=True,
        )
        return

    context_sink = CsvSink(
        output_dir / "cosine_context.csv",
        CONTEXT_FIELDS,
        ("model", "job_id", "layer", "base", "metric_variant"),
        resume=resume,
    )
    diffusion_sink = CsvSink(
        output_dir / "cosine_diffusion.csv",
        DIFFUSION_FIELDS,
        (
            "model",
            "pair_group_id",
            "structured_job_id",
            "control_job_id",
            "layer",
            "metric_variant",
            "distance_bin",
        ),
        resume=resume,
    )
    trajectory_config = context_config.get("trajectory", {})
    trajectory_enabled = (
        isinstance(trajectory_config, Mapping)
        and _as_bool(
            trajectory_config.get("enabled", False),
            "cosine.context.trajectory.enabled",
        )
    )
    trajectory_sink = (
        CsvSink(
            output_dir / "cosine_token_trajectories.csv",
            TRAJECTORY_FIELDS,
            ("model", "job_id", "position", "layer"),
            resume=resume,
        )
        if trajectory_enabled
        else None
    )
    summary_sink = JsonlSink(
        output_dir / "cosine_job_summary.jsonl", resume=resume
    )

    started_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "run_fingerprint": fingerprint,
        "config_path": public_path(config_path, project_root),
        "models_path": public_path(models_path, project_root),
        "manifest_path": public_path(manifest_path, project_root),
        "output_dir": public_path(output_dir, project_root),
        "selected_models": merged_selected,
        "skipped_models": merged_skipped,
        "model_runtime": previous_runtime,
        "selected_jobs": [job["job_id"] for job in manifest_jobs],
        "device": str(device),
        "dtype": dtype_name,
        "resume": resume,
        "definitions": {
            "layers": (
                "layer=0 is the embedding output; layer=k is transformer "
                "layer k for real RNA tokens only"
            ),
            "centered": (
                "each layer is centered by the current sample's mean over all "
                "real RNA tokens"
            ),
            "anisotropy_adjusted": (
                "adjusted cosine = raw cosine - pre-estimated same-layer "
                "random matched-token scalar baseline (nucleotide-matched by "
                "default, configurable); this is not whitening"
            ),
            "same_context_baseline": (
                "pair-count-weighted mean of within-pattern and "
                "within-background same-base cosine"
            ),
            "context_separation": (
                "same_context_baseline_cos - cross_context_same_base_cos"
            ),
            "token_trajectory": (
                "bounded deterministic structured/background token samples; records "
                "adjacent-layer movement, distance to the embedding, same-base "
                "structured-vs-background centroid margin, and an optional fixed "
                "random projection for within-model visualization"
            ),
            "diffusion": (
                "1 - cosine at aligned positions whose nucleotide is unchanged; "
                "changed structured-window positions are excluded and counted"
            ),
            "local_peak": (
                "maximum sampled bin mean below far_field_threshold, falling "
                "back to the full curve if no such bin exists"
            ),
            "far_field": (
                "position-count-weighted sampled diffusion at or beyond "
                "far_field_threshold; null when the sequence has no far field"
            ),
            "leakage_auc": (
                "distance-normalized trapezoidal area under the far-field "
                "diffusion curve; null when no far-field positions exist"
            ),
            "statistical_unit": (
                "job or matched pair-group comparison; sampled token pairs are "
                "not biological replicates"
            ),
        },
        "context_config": context_config,
        "diffusion_config": diffusion_config,
        "trajectory_enabled": trajectory_enabled,
        "requested_max_length": requested_max_length,
    }
    _write_metadata(metadata_path, metadata)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in manifest_jobs:
        grouped[job["pair_group_id"]].append(job)

    try:
        for model_name, model_spec in registry.items():
            pending_groups: list[tuple[str, list[dict[str, Any]], str]] = []
            for group_id, group_jobs in grouped.items():
                selection_signature = hashlib.sha256(
                    "\x1f".join(sorted(job["job_id"] for job in group_jobs)).encode(
                        "utf-8"
                    )
                ).hexdigest()[:12]
                completion_key = (
                    f"group|{fingerprint}|{model_name}|{group_id}|"
                    f"{selection_signature}"
                )
                if completion_key not in summary_sink.completed:
                    pending_groups.append(
                        (group_id, group_jobs, completion_key)
                    )
            if not pending_groups:
                continue

            loaded = load_encoder(
                model_spec,
                device=device,
                dtype=dtype,
                requested_max_length=requested_max_length,
            )
            config = getattr(loaded.model, "config", None)
            metadata.setdefault("model_runtime", {})[model_name] = {
                "backend": model_spec.backend,
                "trained_context_window": model_spec.trained_context_window,
                "checkpoint_config_max_position_embeddings": (
                    loaded.config_max_length
                ),
                "runtime_max_position_embeddings": loaded.runtime_max_length,
                "requested_real_token_length": loaded.effective_max_length,
                "position_embedding_type": getattr(
                    config,
                    "position_embedding_type",
                    model_spec.backend,
                ),
                "weight_dtype": str(next(loaded.model.parameters()).dtype),
                "autocast_dtype": str(loaded.dtype),
                "rope_scaling_policy": model_spec.rope_scaling,
            }
            if model_spec.rope_scaling is not None:
                factors = [
                    runtime_rope_scaling_factor(
                        int(job["length"]), model_spec.rope_scaling
                    )
                    for job in manifest_jobs
                ]
                metadata["model_runtime"][model_name].update(
                    yarn_factor_min=min(factors),
                    yarn_factor_max=max(factors),
                )
            _write_metadata(metadata_path, metadata)
            try:
                for group_id, group_jobs, completion_key in pending_groups:
                    _run_group(
                        loaded,
                        model_spec,
                        group_jobs,
                        context_config,
                        diffusion_config,
                        context_sink,
                        diffusion_sink,
                        trajectory_sink,
                        summary_sink,
                        fingerprint,
                    )
                    summary_sink.append(
                        {
                            "completion_key": completion_key,
                            "record_type": "pair_group_completion",
                            "model": model_name,
                            "pair_group_id": group_id,
                            "selected_job_ids": [
                                job["job_id"] for job in group_jobs
                            ],
                        }
                    )
            finally:
                loaded.close()
                del loaded
                gc.collect()
    finally:
        context_sink.close()
        diffusion_sink.close()
        if trajectory_sink is not None:
            trajectory_sink.close()
        summary_sink.close()

    metadata["status"] = "completed"
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_metadata(metadata_path, metadata)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    run(parse_args(argv))


if __name__ == "__main__":
    main()
