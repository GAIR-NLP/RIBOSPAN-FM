# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generate matched structured/native RNA-context manifests (JSONL, 0-based coords)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .config import (
    ConfigError,
    ExperimentConfig,
    load_experiment_config,
    load_model_registry,
    public_path,
)


DNA_ALPHABET = "ACGT"

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return public_path(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _portable_background_config(config: Mapping[str, Any]) -> Any:
    """Serialize background config so identity hashes ignore clone location."""

    portable = _jsonable(config)
    if isinstance(portable, dict) and "path" in portable:
        portable["path"] = public_path(portable["path"])
    return portable


@dataclass(frozen=True)
class PatternSpec:
    """Serializable metadata for the native-window rearrangement."""

    pattern_id: str
    family: str
    generator: str
    params: Mapping[str, Any]
    description: str

    def __post_init__(self) -> None:
        if not self.pattern_id or not self.family or not self.generator:
            raise ValueError(
                "pattern_id, family, and generator must be non-empty"
            )
        object.__setattr__(
            self, "params", MappingProxyType(dict(self.params))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "family": self.family,
            "generator": self.generator,
            "params": _jsonable(self.params),
            "description": self.description,
        }


class PatternBank:
    """Single-pattern metadata bank used by the current experiment."""

    def __init__(self, patterns: Iterable[PatternSpec]) -> None:
        ordered = tuple(patterns)
        by_id = {pattern.pattern_id: pattern for pattern in ordered}
        if len(by_id) != len(ordered):
            raise ValueError("pattern IDs must be unique")
        self._patterns = ordered
        self._by_id = MappingProxyType(by_id)

    @classmethod
    def default(cls) -> "PatternBank":
        return cls(
            (
                PatternSpec(
                    pattern_id="native_low_complexity",
                    family="composition_conditioned_structure",
                    generator="native_rearrangement",
                    params={
                        "composition": "exact_native_window_counts",
                        "objective": "minimum_transition_complexity",
                    },
                    description=(
                        "Low-complexity rearrangement conditioned on exact "
                        "native window nucleotide counts."
                    ),
                ),
            )
        )

    def select_families(
        self, families: Iterable[str]
    ) -> Tuple[PatternSpec, ...]:
        requested = set(families)
        known = {pattern.family for pattern in self._patterns}
        unknown = requested - known
        if unknown:
            raise ValueError(
                f"unknown pattern families: {sorted(unknown)}"
            )
        return tuple(
            pattern
            for pattern in self._patterns
            if pattern.family in requested
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _stable_id(prefix: str, value: Any, size: int = 20) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:size]}"


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _rng_for(namespace: str, value: Any) -> random.Random:
    payload = f"{namespace}\0{_canonical_json(value)}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def _weighted_base(rng: random.Random, probabilities: Mapping[str, float]) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for base in DNA_ALPHABET:
        cumulative += float(probabilities[base])
        if threshold < cumulative:
            return base
    return "T"


@dataclass(frozen=True)
class BackgroundVariant:
    """One concrete background setting, including expanded GC values."""

    background_type: str
    source_type: str
    config: Mapping[str, Any]
    variant_id: str


@dataclass(frozen=True)
class BackgroundSample:
    """A generated DNA background and provenance metadata."""

    sequence: str
    background_id: str
    metadata: Mapping[str, Any]


class BackgroundFactory:
    """Deterministic synthetic/real background generator with in-memory caching."""

    def __init__(self) -> None:
        self._samples: Dict[Tuple[str, int, int], BackgroundSample] = {}
        self._real_records: Dict[
            Tuple[str, str, str], Tuple[Tuple[str, str], ...]
        ] = {}

    def sample(
        self, variant: BackgroundVariant, length: int, seed: int
    ) -> BackgroundSample:
        """Return a stable sample for one variant, sequence length, and seed."""

        key = (variant.variant_id, length, seed)
        if key not in self._samples:
            if variant.source_type == "synthetic":
                sequence, metadata = self._synthetic(variant, length, seed)
            elif variant.source_type == "real":
                sequence, metadata = self._real(variant, length, seed)
            else:
                raise ValueError(f"unknown source_type {variant.source_type!r}")
            if (
                variant.source_type == "synthetic"
                and len(sequence) != length
            ) or set(sequence) - set(DNA_ALPHABET):
                raise AssertionError(
                    "background generator violated sequence contract"
                )
            identity = {
                "variant_id": variant.variant_id,
                "target_length": length,
                "actual_length": len(sequence),
                "seed": seed,
                "sequence_sha256": _sequence_sha256(sequence),
            }
            self._samples[key] = BackgroundSample(
                sequence=sequence,
                background_id=_stable_id("bg", identity),
                metadata=metadata,
            )
        return self._samples[key]

    def _synthetic(
        self, variant: BackgroundVariant, length: int, seed: int
    ) -> Tuple[str, Dict[str, Any]]:
        config = variant.config
        rng = _rng_for(
            "synthetic-background",
            {"variant": variant.variant_id, "length": length, "seed": seed},
        )
        if variant.background_type == "balanced_iid":
            probabilities = config["probabilities"]
            sequence = "".join(
                _weighted_base(rng, probabilities) for _ in range(length)
            )
        elif variant.background_type == "specified_gc":
            gc = float(config["gc"])
            probabilities = {
                "A": (1.0 - gc) / 2.0,
                "C": gc / 2.0,
                "G": gc / 2.0,
                "T": (1.0 - gc) / 2.0,
            }
            sequence = "".join(
                _weighted_base(rng, probabilities) for _ in range(length)
            )
        elif variant.background_type == "markov":
            initial = config["initial"]
            transitions = config["transitions"]
            current = _weighted_base(rng, initial)
            bases = [current]
            while len(bases) < length:
                current = _weighted_base(rng, transitions[current])
                bases.append(current)
            sequence = "".join(bases)
        else:
            raise ValueError(
                f"unsupported synthetic background {variant.background_type!r}"
            )
        return sequence, {
            "background_variant_id": variant.variant_id,
            "background_config": _jsonable(config),
            "generator": variant.background_type,
        }

    def _load_real_records(
        self, path: Path, id_column: str, sequence_column: str
    ) -> Tuple[Tuple[str, str], ...]:
        cache_key = (str(path), id_column, sequence_column)
        if cache_key in self._real_records:
            return self._real_records[cache_key]
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames is None:
                    raise ValueError(f"real background table {path} has no header")
                missing = {id_column, sequence_column} - set(reader.fieldnames)
                if missing:
                    raise ValueError(
                        f"real background table {path} is missing columns {sorted(missing)}"
                    )
                records = []
                for row_number, row in enumerate(reader, start=2):
                    identifier = (row.get(id_column) or "").strip()
                    raw = "".join((row.get(sequence_column) or "").split()).upper()
                    if not identifier:
                        identifier = f"row-{row_number}"
                    if raw:
                        records.append((identifier, raw))
        except OSError as exc:
            raise ValueError(f"cannot read real background table {path}: {exc}") from exc
        if not records:
            raise ValueError(f"real background table {path} contains no sequences")
        result = tuple(records)
        self._real_records[cache_key] = result
        return result

    def _real(
        self, variant: BackgroundVariant, length: int, seed: int
    ) -> Tuple[str, Dict[str, Any]]:
        config = variant.config
        path = Path(config["path"])
        id_column = str(config["id_column"])
        sequence_column = str(config["sequence_column"])
        records = self._load_real_records(path, id_column, sequence_column)
        tolerance = float(config["length_tolerance_fraction"])
        eligible_indices = [
            index
            for index, (_, sequence) in enumerate(records)
            if abs(len(sequence) - length) <= length * tolerance
            and set(sequence) <= set("ACGTU")
        ]
        if seed >= len(eligible_indices):
            raise ValueError(
                f"length bin {length}±{tolerance:.3%} has only "
                f"{len(eligible_indices)} clean full transcripts, "
                f"cannot select seed index {seed} without replacement"
            )
        order_rng = _rng_for(
            "real-background-order",
            {
                "variant": variant.variant_id,
                "target_length": length,
                "tolerance": tolerance,
            },
        )
        order_rng.shuffle(eligible_indices)
        record_index = eligible_indices[seed]
        identifier, raw_sequence = records[record_index]
        source_ids = [identifier]
        start_offset = 0
        selected = raw_sequence
        extension = {
            "applied": False,
            "policy": "full_transcript_length_bin",
            "source_sequence_length": len(raw_sequence),
            "target_length": length,
            "actual_length": len(raw_sequence),
            "length_tolerance_fraction": tolerance,
        }

        u_count = selected.count("U")
        sequence = selected.replace("U", "T")
        metadata = {
            "background_variant_id": variant.variant_id,
            "background_config": _jsonable(config),
            "generator": "real_mrna",
            "real_background": {
                "source_record_ids": source_ids,
                "start_offset": start_offset,
                "extension": extension,
                "eligible_continuous_records": len(eligible_indices),
                "rna_u_to_t_count": u_count,
                "ambiguous_bases_replaced": 0,
            },
        }
        return sequence, metadata


def _background_variants(
    backgrounds: Mapping[str, Mapping[str, Any]]
) -> Tuple[BackgroundVariant, ...]:
    variants = []
    for background_type, raw_config in backgrounds.items():
        if not raw_config.get("enabled", False):
            continue
        source_type = str(raw_config["source_type"])
        if background_type == "specified_gc":
            for gc_value in raw_config["gc_values"]:
                config = {
                    key: value
                    for key, value in raw_config.items()
                    if key not in {"gc_values", "enabled", "source_type"}
                }
                config["gc"] = float(gc_value)
                identity = {
                    "background_type": background_type,
                    "source_type": source_type,
                    "config": _portable_background_config(config),
                }
                variants.append(
                    BackgroundVariant(
                        background_type=background_type,
                        source_type=source_type,
                        config=config,
                        variant_id=_stable_id("background", identity),
                    )
                )
        else:
            config = {
                key: value
                for key, value in raw_config.items()
                if key not in {"enabled", "source_type"}
            }
            identity = {
                "background_type": background_type,
                "source_type": source_type,
                "config": _portable_background_config(config),
            }
            variants.append(
                BackgroundVariant(
                    background_type=background_type,
                    source_type=source_type,
                    config=config,
                    variant_id=_stable_id("background", identity),
                )
            )
    if not variants:
        raise ValueError("at least one background must be enabled")
    return tuple(variants)


def _strengths(matrix: Mapping[str, Any], length: int) -> Iterator[Tuple[str, Any, int]]:
    mode = str(matrix["pattern_size_mode"])
    if mode == "fixed":
        for pattern_length in matrix["pattern_lengths"]:
            concrete = int(pattern_length)
            if concrete <= length:
                yield "fixed_length", concrete, concrete
        return
    if mode == "fraction":
        for fraction in matrix["pattern_fractions"]:
            concrete = min(
                length, max(1, int(round(length * float(fraction))))
            )
            yield "fraction", float(fraction), concrete
        return
    raise ValueError(f"unsupported pattern_size_mode: {mode}")


def _motif_interval(length: int, motif_length: int, position: float) -> Tuple[int, int]:
    """Place a motif by requested center fraction and clamp it to the sequence."""

    center = int(round(float(position) * (length - 1)))
    start = max(0, min(length - motif_length, center - motif_length // 2))
    return start, start + motif_length


def _small_period_autocorrelation(sequence: str) -> float:
    """Return the strongest short-lag identity; lower is less periodic."""

    if len(sequence) < 2:
        return 1.0
    maximum = 0.0
    max_lag = min(12, len(sequence) // 2)
    for lag in range(1, max_lag + 1):
        overlap = len(sequence) - lag
        identity = sum(
            left == right for left, right in zip(sequence[:-lag], sequence[lag:])
        ) / overlap
        maximum = max(maximum, identity)
    return maximum


def _transition_entropy(sequence: str) -> float:
    """First-order conditional entropy H(next_base | current_base), in bits."""

    if len(sequence) < 2:
        return 0.0
    transitions: Dict[str, Dict[str, int]] = {
        base: {target: 0 for target in DNA_ALPHABET}
        for base in DNA_ALPHABET
    }
    source_counts = {base: 0 for base in DNA_ALPHABET}
    for source, target in zip(sequence[:-1], sequence[1:]):
        transitions[source][target] += 1
        source_counts[source] += 1
    total = len(sequence) - 1
    entropy = 0.0
    for source in DNA_ALPHABET:
        source_count = source_counts[source]
        if not source_count:
            continue
        conditional = 0.0
        for target_count in transitions[source].values():
            if not target_count:
                continue
            probability = target_count / source_count
            conditional -= probability * math.log2(probability)
        entropy += (source_count / total) * conditional
    return entropy


def _transition_count(sequence: str) -> int:
    return sum(left != right for left, right in zip(sequence[:-1], sequence[1:]))


def _max_run_length(sequence: str) -> int:
    if not sequence:
        return 0
    maximum = current = 1
    for left, right in zip(sequence[:-1], sequence[1:]):
        current = current + 1 if left == right else 1
        maximum = max(maximum, current)
    return maximum


def _composition_conditioned_structured(
    native_window: str,
    pair_group_id: str,
) -> Tuple[str, bool, Dict[str, Any]]:
    """Build one deterministic low-complexity native-composition rearrangement."""

    if not native_window:
        raise ValueError("native_window must be non-empty")
    counts = {base: native_window.count(base) for base in DNA_ALPHABET}
    active_bases = [base for base in DNA_ALPHABET if counts[base]]
    if len(active_bases) == 1:
        details = {
            "reason": "native_window_has_one_unique_permutation",
            "composition_preserved": True,
            "native_hamming": 0,
            "transition_entropy": 0.0,
            "transition_count": 0,
            "max_run_length": len(native_window),
        }
        return native_window, True, details

    structured_candidates: List[Tuple[str, int]] = []
    for order in itertools.permutations(active_bases):
        candidate = "".join(base * counts[base] for base in order)
        if candidate == native_window:
            continue
        hamming = sum(
            left != right for left, right in zip(candidate, native_window)
        )
        structured_candidates.append((candidate, hamming))
    if not structured_candidates:
        structured_candidates.append((native_window, 0))

    rng = _rng_for("structured-control", pair_group_id)
    rng.shuffle(structured_candidates)
    best_structured, best_hamming = min(
        structured_candidates,
        key=lambda item: (
            _transition_entropy(item[0]),
            -item[1],
            -_small_period_autocorrelation(item[0]),
        ),
    )
    structured_details = {
        "reason": "composition_preserving_low_complexity_rearrangement",
        "composition_preserved": sorted(best_structured) == sorted(native_window),
        "native_hamming": best_hamming,
        "transition_entropy": _transition_entropy(best_structured),
        "transition_count": _transition_count(best_structured),
        "max_run_length": _max_run_length(best_structured),
        "short_period_autocorrelation": _small_period_autocorrelation(
            best_structured
        ),
    }
    return best_structured, False, structured_details


def _changed_positions(
    sequence: str, reference: str
) -> List[int]:
    if len(sequence) != len(reference):
        raise ValueError("changed-position sequences must have equal lengths")
    return [
        index
        for index, (observed, expected) in enumerate(zip(sequence, reference))
        if observed != expected
    ]


def _records_for_group(
    *,
    pair_group_id: str,
    background: BackgroundSample,
    variant: BackgroundVariant,
    pattern: PatternSpec,
    structured: str,
    structured_degenerate: bool,
    structured_details: Mapping[str, Any],
    control_types: Sequence[str],
    seed: int,
    length: int,
    length_group: int,
    strength_mode: str,
    strength_value: Any,
    motif_start: int,
    motif_end: int,
    requested_position: float,
) -> Iterator[Dict[str, Any]]:
    native_sequence = background.sequence
    for control_type in control_types:
        if control_type == "native":
            sequence = native_sequence
            changed = []
            changed_reference = "native"
            degenerate = False
            control_metadata: Mapping[str, Any] = {
                "reason": "unmodified_shared_native_background"
            }
        else:
            sequence = (
                background.sequence[:motif_start]
                + structured
                + background.sequence[motif_end:]
            )
            reference = native_sequence
            changed_reference = "native"
            degenerate = structured_degenerate
            control_metadata = {
                **dict(structured_details),
                "insertion_noop": sequence == native_sequence,
            }
            changed = _changed_positions(sequence, reference)

        job_identity = {
            "pair_group_id": pair_group_id,
            "control_type": control_type,
        }
        yield {
            "job_id": _stable_id("job", job_identity),
            "pair_group_id": pair_group_id,
            "background_id": background.background_id,
            "seed": seed,
            "length": length,
            "length_group": length_group,
            "source_type": variant.source_type,
            "background_type": variant.background_type,
            "pattern_id": pattern.pattern_id,
            "pattern_family": pattern.family,
            "strength_mode": strength_mode,
            "strength_value": strength_value,
            "motif_start": motif_start,
            "motif_end": motif_end,
            "control_type": control_type,
            "sequence": sequence,
            "sequence_sha256": _sequence_sha256(sequence),
            "changed_positions": changed,
            "degenerate_control": degenerate,
            "metadata": {
                "manifest_schema_version": 3,
                "coordinate_system": "zero_based_half_open",
                "changed_positions_reference": changed_reference,
                "changed_positions_definition": (
                    "absolute zero-based sequence indices differing from the "
                    "record named by changed_positions_reference"
                ),
                "requested_position_fraction": requested_position,
                "actual_pattern_length": motif_end - motif_start,
                "pattern": pattern.to_dict(),
                "background": _jsonable(background.metadata),
                "control": _jsonable(control_metadata),
            },
        }


def iter_pair_records(
    config: ExperimentConfig,
    *,
    pattern_bank: Optional[PatternBank] = None,
    max_groups: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield strict matched-pair records in deterministic group/control order."""

    values = config.values
    matrix = values["matrix"]
    bank = pattern_bank if pattern_bank is not None else PatternBank.default()
    patterns = bank.select_families(values["pattern_families"])
    selected_ids = values.get("pattern_ids")
    if selected_ids is not None:
        selected = set(selected_ids)
        known = {pattern.pattern_id for pattern in patterns}
        unknown = selected - known
        if unknown:
            raise ValueError(
                f"pattern_ids are not enabled by pattern_families: {sorted(unknown)}"
            )
        patterns = tuple(
            pattern for pattern in patterns if pattern.pattern_id in selected
        )
    variants = _background_variants(values["backgrounds"])
    controls = values["controls"]
    control_types = tuple(str(value) for value in controls["types"])
    configured_limit = values["sampling"].get("max_pair_groups")
    if max_groups is None:
        effective_limit = configured_limit
    elif configured_limit is None:
        effective_limit = max_groups
    else:
        effective_limit = min(max_groups, configured_limit)
    if effective_limit is not None and effective_limit < 1:
        raise ValueError("max_groups must be positive or None")

    background_factory = BackgroundFactory()
    emitted_groups = 0
    for length_value in matrix["lengths"]:
        length_group = int(length_value)
        for seed_value in matrix["seeds"]:
            seed = int(seed_value)
            for variant in variants:
                background = background_factory.sample(
                    variant, length_group, seed
                )
                length = len(background.sequence)
                for pattern in patterns:
                    for strength_mode, strength_value, motif_length in _strengths(
                        matrix, length
                    ):
                        for position_value in matrix["positions"]:
                            position = float(position_value)
                            motif_start, motif_end = _motif_interval(
                                length, motif_length, position
                            )
                            group_identity = {
                                "background_id": background.background_id,
                                "seed": seed,
                                "length": length,
                                "length_group": length_group,
                                "background_variant_id": variant.variant_id,
                                "pattern_id": pattern.pattern_id,
                                "strength_mode": strength_mode,
                                "strength_value": strength_value,
                                "motif_length": motif_length,
                                "position": position,
                                "motif_start": motif_start,
                                "motif_end": motif_end,
                            }
                            pair_group_id = _stable_id("pair", group_identity)
                            native_window = background.sequence[
                                motif_start:motif_end
                            ]
                            (
                                structured,
                                structured_degenerate,
                                structured_details,
                            ) = _composition_conditioned_structured(
                                native_window,
                                pair_group_id,
                            )
                            yield from _records_for_group(
                                pair_group_id=pair_group_id,
                                background=background,
                                variant=variant,
                                pattern=pattern,
                                structured=structured,
                                structured_degenerate=structured_degenerate,
                                structured_details=structured_details,
                                control_types=control_types,
                                seed=seed,
                                length=length,
                                length_group=length_group,
                                strength_mode=strength_mode,
                                strength_value=strength_value,
                                motif_start=motif_start,
                                motif_end=motif_end,
                                requested_position=position,
                            )
                            emitted_groups += 1
                            if (
                                effective_limit is not None
                                and emitted_groups >= effective_limit
                            ):
                                return


def generate_manifest(
    config: ExperimentConfig,
    output_path: Optional[Path | str] = None,
    *,
    pattern_bank: Optional[PatternBank] = None,
    max_groups: Optional[int] = None,
) -> Dict[str, Any]:
    """Write a JSONL manifest and return compact generation statistics."""

    destination = (
        Path(output_path).expanduser().resolve(strict=False)
        if output_path is not None
        else config.output_manifest_path
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    pair_groups = set()
    degenerate_controls = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in iter_pair_records(
            config, pattern_bank=pattern_bank, max_groups=max_groups
        ):
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            record_count += 1
            pair_groups.add(record["pair_group_id"])
            degenerate_controls += int(record["degenerate_control"])
    return {
        "output_manifest": str(destination),
        "records": record_count,
        "pair_groups": len(pair_groups),
        "degenerate_controls": degenerate_controls,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="experiment YAML (defaults to configs/experiment.yaml)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="profile name (defaults to default_profile in YAML)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL destination; relative paths are resolved against the project root",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="optional deterministic pair-group cap",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    try:
        config = load_experiment_config(args.config, profile=args.profile)
        load_model_registry(config.models_registry_path)
        output = args.output
        if output is not None and not output.is_absolute():
            output = Path(__file__).resolve().parents[1] / output
        statistics = generate_manifest(
            config, output_path=output, max_groups=args.max_groups
        )
    except (ConfigError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(statistics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BackgroundFactory",
    "BackgroundSample",
    "BackgroundVariant",
    "generate_manifest",
    "iter_pair_records",
    "main",
]
