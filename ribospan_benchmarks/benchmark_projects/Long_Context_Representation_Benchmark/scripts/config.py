# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Experiment YAML loading, profile overlays, and path validation.

Relative paths and ``--config`` are project-root relative.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DNA_ALPHABET = ("A", "C", "G", "T")
REQUIRED_BACKGROUNDS = ("balanced_iid", "specified_gc", "markov", "real_mrna")
REQUIRED_CONTROLS = ("structured", "native")


class ConfigError(ValueError):
    """Raised when an experiment or model registry is malformed."""


@dataclass(frozen=True)
class ModelSpec:
    """One validated model registry entry."""

    name: str
    source_path: Path
    context_window: int


@dataclass(frozen=True)
class ModelRegistry:
    """Validated model registry with source paths resolved to absolute paths."""

    config_path: Path
    models_root: Path
    models: Tuple[ModelSpec, ...]

    def by_name(self) -> Dict[str, ModelSpec]:
        """Return registry entries keyed by canonical model name."""

        return {model.name: model for model in self.models}


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment settings after applying one named profile."""

    config_path: Path
    profile: str
    values: Mapping[str, Any]
    models_registry_path: Path
    output_manifest_path: Path

    @property
    def matrix(self) -> Mapping[str, Any]:
        """Return the experiment matrix section."""

        return self.values["matrix"]

    @property
    def backgrounds(self) -> Mapping[str, Mapping[str, Any]]:
        """Return enabled and disabled background specifications."""

        return self.values["backgrounds"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_experiment_path() -> Path:
    return _project_root() / "configs" / "experiment.yaml"


def public_path(path: Path | str, root: Path | None = None) -> str:
    """Return a repository-relative path for metadata JSON."""

    value = Path(path)
    base = Path(root) if root is not None else _project_root()
    try:
        resolved = value.resolve() if value.is_absolute() else (base / value).resolve()
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping with PyYAML's non-executing safe loader."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigError("PyYAML is required to read experiment configuration") from exc

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read YAML file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a top-level mapping")
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings while replacing scalar values and lists."""

    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_relative(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{field} must be a sequence")
    if not value:
        raise ConfigError(f"{field} must not be empty")
    return value


def _unique_numeric(
    value: Any,
    field: str,
    *,
    integral: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Tuple[float, ...]:
    items = _require_sequence(value, field)
    converted = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{field} contains a non-numeric value: {item!r}")
        number = float(item)
        if integral and not number.is_integer():
            raise ConfigError(f"{field} must contain integers")
        if minimum is not None and number < minimum:
            raise ConfigError(f"{field} values must be >= {minimum}")
        if maximum is not None and number > maximum:
            raise ConfigError(f"{field} values must be <= {maximum}")
        converted.append(number)
    if len(set(converted)) != len(converted):
        raise ConfigError(f"{field} must not contain duplicates")
    return tuple(converted)


def _validate_distribution(value: Any, field: str) -> None:
    distribution = _require_mapping(value, field)
    if set(distribution) != set(DNA_ALPHABET):
        raise ConfigError(f"{field} must define exactly A, C, G, and T")
    probabilities = []
    for base in DNA_ALPHABET:
        probability = distribution[base]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ConfigError(f"{field}.{base} must be numeric")
        if float(probability) < 0.0:
            raise ConfigError(f"{field}.{base} must be non-negative")
        probabilities.append(float(probability))
    if abs(sum(probabilities) - 1.0) > 1e-8:
        raise ConfigError(f"{field} probabilities must sum to 1")


def _validate_experiment(values: Mapping[str, Any], profile: str) -> None:
    if values.get("schema_version") != 1:
        raise ConfigError("experiment schema_version must be 1")

    matrix = _require_mapping(values.get("matrix"), "matrix")
    lengths = _unique_numeric(matrix.get("lengths"), "matrix.lengths", integral=True, minimum=1)
    pattern_size_mode = matrix.get("pattern_size_mode")
    if pattern_size_mode not in {"fixed", "fraction"}:
        raise ConfigError(
            "matrix.pattern_size_mode must be fixed or fraction"
        )
    pattern_lengths_value = matrix.get("pattern_lengths")
    pattern_fractions_value = matrix.get("pattern_fractions")
    if pattern_size_mode == "fixed":
        pattern_lengths = _unique_numeric(
            pattern_lengths_value,
            "matrix.pattern_lengths",
            integral=True,
            minimum=1,
        )
        if pattern_fractions_value != []:
            raise ConfigError(
                "matrix.pattern_fractions must be empty in fixed mode"
            )
        if min(pattern_lengths) > max(lengths):
            raise ConfigError(
                "all fixed pattern lengths exceed all sequence lengths"
            )
    else:
        if pattern_lengths_value != []:
            raise ConfigError(
                "matrix.pattern_lengths must be empty in fraction mode"
            )
        pattern_fractions = _unique_numeric(
            pattern_fractions_value,
            "matrix.pattern_fractions",
            minimum=0.0,
            maximum=1.0,
        )
        if any(value <= 0.0 for value in pattern_fractions):
            raise ConfigError(
                "matrix.pattern_fractions values must be greater than 0"
            )
    _unique_numeric(
        matrix.get("positions"), "matrix.positions", minimum=0.0, maximum=1.0
    )
    seeds = _unique_numeric(matrix.get("seeds"), "matrix.seeds", integral=True, minimum=0)
    if profile == "publication" and len(seeds) < 50:
        raise ConfigError(f"profile {profile!r} must define at least 50 seeds")

    families = _require_sequence(values.get("pattern_families"), "pattern_families")
    if len(set(families)) != len(families) or not all(isinstance(x, str) for x in families):
        raise ConfigError("pattern_families must contain unique strings")
    pattern_ids = values.get("pattern_ids")
    if pattern_ids is not None:
        selected_patterns = _require_sequence(pattern_ids, "pattern_ids")
        if len(set(selected_patterns)) != len(selected_patterns) or not all(
            isinstance(item, str) and item for item in selected_patterns
        ):
            raise ConfigError("pattern_ids must be null or contain unique non-empty strings")

    backgrounds = _require_mapping(values.get("backgrounds"), "backgrounds")
    missing_backgrounds = set(REQUIRED_BACKGROUNDS) - set(backgrounds)
    if missing_backgrounds:
        raise ConfigError(f"missing required backgrounds: {sorted(missing_backgrounds)}")
    for name in REQUIRED_BACKGROUNDS:
        background = _require_mapping(backgrounds[name], f"backgrounds.{name}")
        if not isinstance(background.get("enabled"), bool):
            raise ConfigError(f"backgrounds.{name}.enabled must be boolean")
        expected_source = "real" if name == "real_mrna" else "synthetic"
        if background.get("source_type") != expected_source:
            raise ConfigError(
                f"backgrounds.{name}.source_type must be {expected_source!r}"
            )
    _validate_distribution(
        backgrounds["balanced_iid"].get("probabilities"),
        "backgrounds.balanced_iid.probabilities",
    )
    _unique_numeric(
        backgrounds["specified_gc"].get("gc_values"),
        "backgrounds.specified_gc.gc_values",
        minimum=0.0,
        maximum=1.0,
    )
    markov = backgrounds["markov"]
    _validate_distribution(markov.get("initial"), "backgrounds.markov.initial")
    transitions = _require_mapping(
        markov.get("transitions"), "backgrounds.markov.transitions"
    )
    if set(transitions) != set(DNA_ALPHABET):
        raise ConfigError("backgrounds.markov.transitions must have A/C/G/T rows")
    for base in DNA_ALPHABET:
        _validate_distribution(
            transitions[base], f"backgrounds.markov.transitions.{base}"
        )
    real_mrna = backgrounds["real_mrna"]
    if not isinstance(real_mrna.get("path"), (str, Path)) or not real_mrna["path"]:
        raise ConfigError("backgrounds.real_mrna.path must be a path")
    for field in ("id_column", "sequence_column"):
        if not isinstance(real_mrna.get(field), str) or not real_mrna[field]:
            raise ConfigError(f"backgrounds.real_mrna.{field} must be a string")
    if real_mrna.get("short_sequence_policy") != "length_bin_full":
        raise ConfigError(
            "backgrounds.real_mrna.short_sequence_policy must be "
            "length_bin_full"
        )
    tolerance = real_mrna.get("length_tolerance_fraction")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not 0.0 < float(tolerance) < 1.0
    ):
        raise ConfigError(
            "backgrounds.real_mrna.length_tolerance_fraction must be "
            "in (0, 1)"
        )

    controls = _require_mapping(values.get("controls"), "controls")
    control_types = _require_sequence(controls.get("types"), "controls.types")
    if tuple(control_types) != REQUIRED_CONTROLS:
        raise ConfigError(
            "controls.types must be ordered as structured, native"
        )

    sampling = _require_mapping(values.get("sampling"), "sampling")
    bins = _unique_numeric(
        sampling.get("representation_distance_bins"),
        "sampling.representation_distance_bins",
        integral=True,
        minimum=0.0,
    )
    if any(left >= right for left, right in zip(bins, bins[1:])):
        raise ConfigError(
            "sampling.representation_distance_bins must be strictly increasing"
        )
    for field in (
        "far_field_threshold",
        "anisotropy_pair_sample_size",
        "context_pair_sample_size",
        "max_positions_per_bin",
    ):
        value = sampling.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"sampling.{field} must be a positive integer")
    max_groups = sampling.get("max_pair_groups")
    if max_groups is not None and (
        not isinstance(max_groups, int) or isinstance(max_groups, bool) or max_groups < 1
    ):
        raise ConfigError("sampling.max_pair_groups must be null or a positive integer")

    attention = _require_mapping(values.get("attention"), "attention")
    if not isinstance(attention.get("enabled"), bool):
        raise ConfigError("attention.enabled must be boolean")
    if not isinstance(attention.get("batch_size"), int) or attention["batch_size"] < 1:
        raise ConfigError("attention.batch_size must be a positive integer")
    for field in ("heatmap_window", "query_chunk_size"):
        value = attention.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"attention.{field} must be a positive integer")
    pool_size = attention.get("heatmap_pool_size")
    if (
        not isinstance(pool_size, int)
        or isinstance(pool_size, bool)
        or pool_size < 0
    ):
        raise ConfigError(
            "attention.heatmap_pool_size must be zero or a positive integer"
        )
    norm_quantile = attention.get("heatmap_norm_quantile")
    if (
        not isinstance(norm_quantile, (int, float))
        or isinstance(norm_quantile, bool)
        or not 0.0 < float(norm_quantile) <= 1.0
    ):
        raise ConfigError("attention.heatmap_norm_quantile must be in (0, 1]")

    cosine = _require_mapping(values.get("cosine"), "cosine")
    context = _require_mapping(cosine.get("context"), "cosine.context")
    trajectory = context.get("trajectory")
    if trajectory is not None:
        trajectory = _require_mapping(
            trajectory, "cosine.context.trajectory"
        )
        if not isinstance(trajectory.get("enabled"), bool):
            raise ConfigError(
                "cosine.context.trajectory.enabled must be boolean"
            )
        positions = trajectory.get("positions_per_base_region")
        if (
            not isinstance(positions, int)
            or isinstance(positions, bool)
            or positions < 1
        ):
            raise ConfigError(
                "cosine.context.trajectory.positions_per_base_region "
                "must be a positive integer"
            )
        projection_dim = trajectory.get("projection_dim")
        if (
            not isinstance(projection_dim, int)
            or isinstance(projection_dim, bool)
            or not 0 <= projection_dim <= 3
        ):
            raise ConfigError(
                "cosine.context.trajectory.projection_dim must be in [0, 3]"
            )
        projection_seed = trajectory.get("projection_seed")
        if (
            not isinstance(projection_seed, int)
            or isinstance(projection_seed, bool)
            or projection_seed < 0
        ):
            raise ConfigError(
                "cosine.context.trajectory.projection_seed must be "
                "a non-negative integer"
            )


def load_experiment_config(
    path: Optional[Path | str] = None, profile: Optional[str] = None
) -> ExperimentConfig:
    """Load, overlay, resolve, and validate an experiment configuration.

    Args:
        path: Experiment YAML path.  Defaults to ``configs/experiment.yaml``.
        profile: Named profile to apply.  Defaults to ``default_profile``.
    """

    config_path = (
        Path(path).expanduser()
        if path is not None
        else _default_experiment_path()
    )
    if not config_path.is_absolute():
        config_path = _project_root() / config_path
    config_path = config_path.resolve(strict=False)
    raw = _load_yaml(config_path)
    profiles = _require_mapping(raw.get("profiles"), "profiles")
    selected = profile if profile is not None else raw.get("default_profile")
    if not isinstance(selected, str) or selected not in profiles:
        raise ConfigError(
            f"unknown profile {selected!r}; available profiles: {sorted(profiles)}"
        )
    overlay = _require_mapping(profiles[selected], f"profiles.{selected}")
    base = {key: value for key, value in raw.items() if key != "profiles"}
    values = _deep_merge(base, overlay)

    root = _project_root()
    models_path = _resolve_relative(root, values.get("models_registry"), "models_registry")
    output_path = _resolve_relative(
        root, values.get("output_manifest"), "output_manifest"
    )
    backgrounds = dict(_require_mapping(values.get("backgrounds"), "backgrounds"))
    real_mrna = dict(
        _require_mapping(backgrounds.get("real_mrna"), "backgrounds.real_mrna")
    )
    real_mrna["path"] = _resolve_relative(
        root, real_mrna.get("path"), "backgrounds.real_mrna.path"
    )
    backgrounds["real_mrna"] = real_mrna
    values["backgrounds"] = backgrounds
    values["models_registry"] = models_path
    values["output_manifest"] = output_path
    values["active_profile"] = selected

    _validate_experiment(values, selected)
    return ExperimentConfig(
        config_path=config_path,
        profile=selected,
        values=values,
        models_registry_path=models_path,
        output_manifest_path=output_path,
    )


def load_model_registry(path: Optional[Path | str] = None) -> ModelRegistry:
    """Load and validate the single canonical model registry.

    Missing checkpoint directories are allowed so a registry can be prepared
    before model materialization.  Consumers that load weights should check the
    resolved ``source_path`` at that boundary.
    """

    registry_path = (
        Path(path).expanduser().resolve(strict=False)
        if path is not None
        else _project_root() / "configs" / "models.yaml"
    )
    raw = _load_yaml(registry_path)
    if raw.get("schema_version") != 1:
        raise ConfigError("model registry schema_version must be 1")
    models_root = _resolve_relative(
        registry_path.parent, raw.get("models_root"), "models_root"
    )
    entries = _require_sequence(raw.get("models"), "models")
    required_fields = {"name", "source_path", "context_window"}
    models = []
    raw_names = []
    for index, entry_value in enumerate(entries):
        entry = _require_mapping(entry_value, f"models[{index}]")
        missing = required_fields - set(entry)
        if missing:
            raise ConfigError(f"models[{index}] is missing fields: {sorted(missing)}")
        name = entry["name"]
        if not isinstance(name, str):
            raise ConfigError(f"models[{index}].name must be a string")
        source_value = entry["source_path"]
        if not isinstance(source_value, str) or Path(source_value).is_absolute():
            raise ConfigError(
                f"models[{index}].source_path must be relative to models_root"
            )
        context_window = entry["context_window"]
        if (
            not isinstance(context_window, int)
            or isinstance(context_window, bool)
            or context_window < 1
        ):
            raise ConfigError(f"models[{index}].context_window must be positive")
        raw_names.append(name)
        models.append(
            ModelSpec(
                name=name,
                source_path=(models_root / source_value).resolve(strict=False),
                context_window=context_window,
            )
        )

    if not raw_names:
        raise ConfigError("model registry must contain at least one model")
    if len(set(raw_names)) != len(raw_names):
        raise ConfigError("model names must be unique")
    return ModelRegistry(
        config_path=registry_path, models_root=models_root, models=tuple(models)
    )


__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "ModelRegistry",
    "ModelSpec",
    "load_experiment_config",
    "load_model_registry",
    "public_path",
]
