# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Local HuggingFace encoder registry loading and hidden-state I/O."""

from __future__ import annotations

import gc
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from scripts.model_src import ensure_import_path


SUPPORTED_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


HF_DIR_BACKENDS = ("rnabert", "ribospan")


def infer_hf_backend(path: Path) -> str:
    """Infer ``rnabert`` vs ``ribospan`` from a local HuggingFace ``config.json``."""

    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if not config_path.is_file():
        return "rnabert"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "rnabert"
    model_type = str(payload.get("model_type") or "").strip().lower()
    if model_type == "ribospan":
        return "ribospan"
    return "rnabert"


def hf_encoder(model: Any) -> Any:
    """Return the transformer body (``model.ribospan`` or ``model.bert``)."""

    encoder = getattr(model, "ribospan", None)
    if encoder is not None:
        return encoder
    encoder = getattr(model, "bert", None)
    if encoder is not None:
        return encoder
    raise AttributeError(
        f"{type(model).__name__} has neither a .ribospan nor a .bert encoder"
    )


@dataclass(frozen=True)
class ModelSpec:
    """One model registry entry."""

    name: str
    path: Path
    trained_context_window: int
    backend: str = "rnabert"
    rope_scaling: dict[str, Any] | None = None


@dataclass
class LoadedRNABert:
    """A local HuggingFace MaskedLM encoder plus tokenizer and length limits."""

    spec: ModelSpec
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    config_max_length: int
    runtime_max_length: int
    effective_max_length: int
    rope_scaling: dict[str, Any] | None = None

    @property
    def num_hidden_layers(self) -> int:
        """Number of transformer layers, excluding the embedding layer."""

        return int(self.model.config.num_hidden_layers)

    def hidden_states(self, sequence: str) -> tuple[torch.Tensor, ...]:
        """Return embedding + every transformer layer for real RNA tokens.

        The sequence is mapped one nucleotide at a time and wrapped as
        ``[CLS] nucleotide... [SEP]``.  Special-token representations are
        removed.  Returned tensors are detached float32 CPU tensors so metric
        accumulation is stable and GPU memory can be released promptly.
        """

        normalized = normalize_sequence(sequence)
        if len(normalized) > self.effective_max_length:
            raise ValueError(
                f"{self.spec.name}: RNA length={len(normalized)} exceeds "
                f"requested real-token maximum={self.effective_max_length}"
            )

        configure_runtime_rope_scaling(
            self.model,
            real_sequence_length=len(normalized),
            policy=self.rope_scaling,
        )

        token_ids = [self.tokenizer.token_to_id(base) for base in normalized]
        unknown_id = self.tokenizer.unk_token_id
        for position, (base, token_id) in enumerate(zip(normalized, token_ids)):
            if token_id == unknown_id and base != "N":
                raise ValueError(
                    f"Tokenizer cannot map base {base!r} at sequence position {position}"
                )
        input_ids = torch.tensor(
            [[self.tokenizer.cls_token_id, *token_ids, self.tokenizer.sep_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Call the copied backbone only (no MLM head / logits).
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.dtype != torch.float32
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            outputs = hf_encoder(self.model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states
        expected = self.num_hidden_layers + 1
        if hidden_states is None or len(hidden_states) != expected:
            actual = None if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"{self.spec.name}: expected {expected} hidden states "
                f"(embedding + layers), got {actual}"
            )
        real_hidden = tuple(
            state[0, 1:-1].detach().to(device="cpu", dtype=torch.float32)
            for state in hidden_states
        )
        if any(state.shape[0] != len(normalized) for state in real_hidden):
            raise RuntimeError("Special-token removal produced an unexpected length")
        del outputs, hidden_states, input_ids, attention_mask
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return real_hidden

    def close(self) -> None:
        """Release model memory before another registry model is loaded."""

        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def embed_pooled(
        self,
        sequence: str,
        *,
        layers: list[int],
        poolings: list[str],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Return pooled vectors keyed by layer then pooling name."""
        from scripts.backends import _pool_cls_body

        normalized = normalize_sequence(sequence)
        if len(normalized) > self.effective_max_length:
            raise ValueError(
                f"{self.spec.name}: RNA length={len(normalized)} exceeds "
                f"requested real-token maximum={self.effective_max_length}"
            )

        configure_runtime_rope_scaling(
            self.model,
            real_sequence_length=len(normalized),
            policy=self.rope_scaling,
        )

        token_ids = [self.tokenizer.token_to_id(base) for base in normalized]
        input_ids = torch.tensor(
            [[self.tokenizer.cls_token_id, *token_ids, self.tokenizer.sep_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.dtype != torch.float32
            else nullcontext()
        )
        with autocast_context:
            outputs = hf_encoder(self.model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states
        vectors: dict[int, dict[str, np.ndarray]] = {}
        for layer in layers:
            layer_vecs: dict[str, np.ndarray] = {}
            for pooling in poolings:
                pooled = _pool_cls_body(hidden_states[layer][0], pooling=pooling)
                layer_vecs[pooling] = (
                    pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
                )
            vectors[layer] = layer_vecs
        del outputs, hidden_states, input_ids, attention_mask
        return vectors

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


def normalize_rope_scaling_policy(value: Any) -> dict[str, Any] | None:
    """Validate an optional inference-time RoPE scaling policy."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("rope_scaling must be a mapping or null")
    scaling_type = str(value.get("type", "")).lower()
    if scaling_type != "yarn":
        raise ValueError("Only rope_scaling.type=yarn is supported")
    factor_mode = str(value.get("factor_mode", "fixed")).lower()
    if factor_mode not in {"fixed", "sequence_length"}:
        raise ValueError(
            "rope_scaling.factor_mode must be fixed or sequence_length"
        )
    original_max = int(value.get("original_max_position_embeddings", 0))
    if original_max < 1:
        raise ValueError(
            "rope_scaling.original_max_position_embeddings must be positive"
        )
    special_tokens = int(value.get("special_tokens", 2))
    if special_tokens < 0:
        raise ValueError("rope_scaling.special_tokens must be non-negative")
    alpha = float(value.get("yarn_alpha", 1.0))
    beta = float(value.get("yarn_beta", 4.0))
    if not math.isfinite(alpha) or not math.isfinite(beta) or beta <= alpha:
        raise ValueError("rope_scaling requires finite yarn_beta > yarn_alpha")
    policy: dict[str, Any] = {
        "type": "yarn",
        "factor_mode": factor_mode,
        "original_max_position_embeddings": original_max,
        "special_tokens": special_tokens,
        "yarn_alpha": alpha,
        "yarn_beta": beta,
    }
    if factor_mode == "fixed":
        factor = float(value.get("factor", 0.0))
        if not math.isfinite(factor) or factor < 1.0:
            raise ValueError("fixed rope_scaling.factor must be finite and >= 1")
        policy["factor"] = factor
    return policy


def runtime_rope_scaling_factor(
    real_sequence_length: int,
    policy: Mapping[str, Any],
) -> float:
    """Resolve the YaRN factor for one real RNA sequence length."""

    normalized = normalize_rope_scaling_policy(policy)
    if normalized is None:
        raise ValueError("A rope_scaling policy is required")
    if normalized["factor_mode"] == "sequence_length":
        total = int(real_sequence_length) + int(normalized["special_tokens"])
        original = int(normalized["original_max_position_embeddings"])
        return max(1.0, float(total) / float(original))
    return float(normalized["factor"])


def configure_runtime_rope_scaling(
    model: Any,
    *,
    real_sequence_length: int,
    policy: Mapping[str, Any] | None,
) -> float | None:
    """Apply fixed or input-length-adaptive YaRN before one model forward."""

    normalized = normalize_rope_scaling_policy(policy)
    if normalized is None:
        return None
    total_positions = int(real_sequence_length) + int(normalized["special_tokens"])
    original_max = int(normalized["original_max_position_embeddings"])
    factor = runtime_rope_scaling_factor(real_sequence_length, normalized)
    backbone = hf_encoder(model)
    rotary = getattr(backbone, "rotary_pos_emb", None)
    if rotary is None:
        raise ValueError("YaRN policy requires a RoPE encoder")
    alpha = float(normalized["yarn_alpha"])
    beta = float(normalized["yarn_beta"])
    current = getattr(model, "_benchmark_current_yarn_factor", None)
    cached_length = int(getattr(rotary, "max_seq_len_cached", 0))
    if (
        current is None
        or not math.isclose(float(current), factor)
        or cached_length != total_positions
    ):
        rotary.scaling_factor = factor
        rotary.original_max_position_embeddings = original_max
        rotary.yarn_alpha = alpha
        rotary.yarn_beta = beta
        rotary._set_cos_sin_cache(seq_len=total_positions)
        inv_temp = 0.1 * math.log(factor) + 1.0
        temperature = 1.0 / (inv_temp ** 2)
        for layer in backbone.encoder.layer:
            layer.attention.temperature = temperature
            layer.attention.self.temperature = temperature
        model._benchmark_current_yarn_factor = factor
        model._benchmark_current_yarn_total_positions = total_positions
        model._benchmark_current_yarn_temperature = temperature
    resolved = {
        "type": "yarn",
        "factor": factor,
        "factor_mode": normalized["factor_mode"],
        "original_max_position_embeddings": original_max,
        "yarn_alpha": alpha,
        "yarn_beta": beta,
    }
    model.config.rope_scaling = resolved
    backbone.config.rope_scaling = resolved
    return factor


def checkpoint_files_present(path: Path, backend: str | None = "rnabert") -> tuple[bool, str]:
    """Return whether ``path`` has loadable weights (empty dirs count as missing)."""

    name = (backend or "rnabert").lower()
    if name in HF_DIR_BACKENDS:
        if not path.is_dir():
            return False, f"missing {name} checkpoint directory {path}"
        has_config = (path / "config.json").is_file()
        has_weights = (
            (path / "pytorch_model.bin").is_file()
            or (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin.index.json").is_file()
            or any(path.glob("pytorch_model-*.bin"))
            or any(path.glob("*.safetensors"))
        )
        if has_config and has_weights:
            return True, ""
        return False, f"missing {name} weights under {path}"
    if name in {"rnafm", "rna-fm", "rinalmo", "rinalmo-giga"}:
        if path.is_file() and path.suffix in {".pt", ".pth"}:
            return True, ""
        if path.is_dir() and (
            any(path.glob("*.pt")) or any(path.glob("*.pth"))
        ):
            return True, ""
        return False, f"missing {name} weights under {path}"
    if name in {"hydrarna", "hydra-rna", "hydra"}:
        if path.is_file() and path.suffix == ".pt":
            return True, ""
        if path.is_dir() and (
            any(
                (path / cand).is_file()
                for cand in (
                    "HydraRNA_model.pt",
                    "hydrarna.pt",
                    "checkpoint_best.pt",
                )
            )
            or any(path.glob("*.pt"))
        ):
            return True, ""
        return False, f"missing HydraRNA checkpoint under {path}"
    return True, ""


def weights_available(spec: ModelSpec) -> tuple[bool, str]:
    return checkpoint_files_present(spec.path, spec.backend)


def normalize_sequence(sequence: str) -> str:
    """Normalize U to T and validate direct single-nucleotide tokenization."""

    normalized = sequence.strip().upper().replace("U", "T")
    if not normalized:
        raise ValueError("Sequence must not be empty")
    invalid = sorted(set(normalized) - {"A", "C", "G", "T", "N"})
    if invalid:
        raise ValueError(f"Unsupported nucleotide symbols: {invalid}")
    return normalized


def dtype_from_name(name: str) -> torch.dtype:
    """Resolve one of the explicitly supported inference dtypes."""

    try:
        return SUPPORTED_DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose from {sorted(SUPPORTED_DTYPES)}"
        ) from exc


def resolve_device(name: str | None) -> torch.device:
    """Resolve CPU/CUDA and fail early for unavailable CUDA requests."""

    if name is None or name.lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({device}) but CUDA is unavailable")
        torch.cuda.set_device(device)
    return device


def _registry_entries(document: Any) -> Sequence[Any] | Mapping[str, Any]:
    if isinstance(document, Mapping) and "models" in document:
        entries = document["models"]
    else:
        entries = document
    if not isinstance(entries, (Mapping, list)):
        raise ValueError("models.yaml must contain a mapping or list of models")
    return entries


def load_model_registry(
    registry_path: Path,
    *,
    config_dir: Path,
    project_root: Path,
) -> dict[str, ModelSpec]:
    """Read flexible mapping/list forms from models.yaml.

    Accepted mapping examples are ``name: path`` and
    ``name: {path: ..., max_length: ...}``.  A list may contain objects with
    ``name`` and ``path``.  Relative checkpoint paths are resolved against the
    run config directory first, then the benchmark project root.
    """

    with registry_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    entries = _registry_entries(document)
    models_root_value = (
        document.get("models_root", ".") if isinstance(document, Mapping) else "."
    )
    models_root = Path(models_root_value).expanduser()
    if not models_root.is_absolute():
        models_root = registry_path.parent / models_root
    models_root = models_root.resolve()
    parsed: list[tuple[str, Any]] = []
    if isinstance(entries, Mapping):
        parsed = [(str(name), value) for name, value in entries.items()]
    else:
        for index, value in enumerate(entries):
            if not isinstance(value, Mapping) or "name" not in value:
                raise ValueError(
                    f"models.yaml list entry {index} must have name and path"
                )
            parsed.append((str(value["name"]), value))

    registry: dict[str, ModelSpec] = {}
    for name, value in parsed:
        if not name:
            raise ValueError("Model names must not be empty")
        if isinstance(value, (str, Path)):
            raw_path = value
            raw_max = None
            raw_rope_scaling = None
            declared_backend = None
        elif isinstance(value, Mapping):
            raw_path = value.get(
                "source_path", value.get("path", value.get("model_path"))
            )
            raw_max = value.get("context_window")
            raw_rope_scaling = value.get("rope_scaling")
            declared_backend = value.get("backend")
        else:
            raise ValueError(f"Invalid registry entry for model {name!r}")
        if raw_path is None:
            raise ValueError(f"Registry model {name!r} has no path")
        model_path_value = Path(raw_path).expanduser()
        if model_path_value.is_absolute():
            model_path = model_path_value.resolve()
        else:
            model_path = (models_root / model_path_value).resolve()
        if declared_backend is None or str(declared_backend).strip() == "":
            raw_backend = infer_hf_backend(model_path)
        else:
            raw_backend = str(declared_backend).strip().lower()
        trained_context_window = int(raw_max) if raw_max is not None else 0
        if trained_context_window < 1:
            raise ValueError(
                f"Model {name!r} must declare a positive context_window"
            )
        if name in registry:
            raise ValueError(f"Duplicate registry model name: {name}")
        registry[name] = ModelSpec(
            name=name,
            path=model_path,
            trained_context_window=trained_context_window,
            backend=raw_backend,
            rope_scaling=normalize_rope_scaling_policy(raw_rope_scaling),
        )
    if not registry:
        raise ValueError("models.yaml contains no models")
    return registry


def load_hf_mlm_checkpoint(
    path: Path,
    *,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    requested_real_tokens: int,
    model_name: str = "",
) -> tuple[Any, Any, int, int]:
    """Load a bundled RNABert or RiboSpan MaskedLM checkpoint.

    Returns ``(model, tokenizer, config_max, runtime_max)``.
    """

    name = (backend or "rnabert").lower()
    if name not in HF_DIR_BACKENDS:
        raise ValueError(f"Unsupported HuggingFace backend {backend!r}")
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 autocast is not supported for this CPU benchmark")
    ensure_import_path()
    if name == "ribospan":
        import ribospan as package
        from ribospan import RiboSpanConfig as Config
        from ribospan import RiboSpanForMaskedLM as Model
        from ribospan import RiboSpanTokenizer as Tokenizer
    else:
        import rnabert as package
        from rnabert import RNABertConfig as Config
        from rnabert import RNABertForMaskedLM as Model
        from rnabert import RNABertTokenizer as Tokenizer

    local_vocab = Path(package.__file__).resolve().parent / "vocab.txt"
    tokenizer = Tokenizer(str(local_vocab))
    checkpoint_config = Config.from_pretrained(str(path), local_files_only=True)
    config_max = int(checkpoint_config.max_position_embeddings)
    runtime_max = max(config_max, int(requested_real_tokens) + 2)
    if (
        runtime_max > config_max
        and checkpoint_config.position_embedding_type != "rope"
    ):
        label = model_name or str(path)
        raise ValueError(
            f"{label}: runtime extension from {config_max} to {runtime_max} "
            "is only supported for RoPE checkpoints"
        )
    checkpoint_config.max_position_embeddings = runtime_max
    model = Model.from_pretrained(
        str(path),
        config=checkpoint_config,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    return model, tokenizer, config_max, runtime_max


def load_hf_masked_lm(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
    backend: str | None = None,
) -> LoadedRNABert:
    """Load RNABertForMaskedLM or RiboSpanForMaskedLM from ``spec.path``."""

    resolved_backend = (backend or spec.backend or "rnabert").lower()
    requested_real_tokens = (
        requested_max_length
        if requested_max_length is not None
        else spec.trained_context_window
    )
    model, tokenizer, config_max, runtime_max = load_hf_mlm_checkpoint(
        spec.path,
        backend=resolved_backend,
        device=device,
        dtype=dtype,
        requested_real_tokens=int(requested_real_tokens),
        model_name=spec.name,
    )
    return LoadedRNABert(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        config_max_length=config_max,
        runtime_max_length=runtime_max,
        effective_max_length=int(requested_real_tokens),
        rope_scaling=spec.rope_scaling,
    )


def load_rnabert(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedRNABert:
    """Load AIDO-style RNABertForMaskedLM weights."""

    return load_hf_masked_lm(
        spec,
        device=device,
        dtype=dtype,
        requested_max_length=requested_max_length,
        backend="rnabert",
    )


def load_ribospan(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_max_length: int | None = None,
) -> LoadedRNABert:
    """Load published RiboSpanForMaskedLM weights."""

    return load_hf_masked_lm(
        spec,
        device=device,
        dtype=dtype,
        requested_max_length=requested_max_length,
        backend="ribospan",
    )
