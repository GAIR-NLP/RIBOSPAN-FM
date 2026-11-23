# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Masked-nucleotide accuracy on the native transcripts of the pair manifest."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import load_experiment_config, public_path
from .cosine import (
    CsvSink,
    _as_bool,
    _config_path,
    _file_digest,
    _job_metric_seed,
    _project_root,
    _resolve_input_path,
    _resolve_output_path,
    load_manifest,
)
from .model_io import (
    HF_DIR_BACKENDS,
    ModelSpec,
    backend_runtime_available,
    configure_runtime_rope_scaling,
    dtype_from_name,
    load_encoder,
    load_model_registry,
    normalize_sequence,
    resolve_device,
    weights_available,
)


NUCLEOTIDES = ("A", "C", "G", "T")
MLM_FIELDS = [
    "model",
    "model_path",
    "job_id",
    "pair_group_id",
    "seed",
    "length",
    "length_group",
    "mask_rate",
    "masked_positions",
    "mlm_accuracy",
]
MLM_DEFINITIONS = {
    "masked_positions": (
        "round(mask_rate * length) positions per native transcript, drawn once "
        "from the job seed and shared by every model"
    ),
    "mlm_accuracy": (
        "every selected position is replaced by the mask token; accuracy is the "
        "fraction whose argmax over the A, C, G, T logits is the original "
        "nucleotide"
    ),
}


@dataclass
class StageRun:
    """Resolved inputs shared by the mlm and restricted_attention stages."""

    project_root: Path
    config: dict[str, Any]
    config_path: Path
    models_path: Path
    manifest_path: Path
    output_dir: Path
    requested_models: list[str]
    runnable: dict[str, ModelSpec]
    skipped_models: list[dict[str, str]]
    manifest_jobs: list[dict[str, Any]]
    requested_max_length: int
    mask_rate: float
    dtype_name: str
    dtype: torch.dtype
    device: torch.device
    resume: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "config_path": public_path(self.config_path, self.project_root),
            "models_path": public_path(self.models_path, self.project_root),
            "manifest_path": public_path(self.manifest_path, self.project_root),
            "output_dir": public_path(self.output_dir, self.project_root),
            "skipped_models": self.skipped_models,
            "device": str(self.device),
            "dtype": self.dtype_name,
            "resume": self.resume,
            "mask_rate": self.mask_rate,
        }

    def fingerprint(self) -> str:
        payload = {
            "manifest_sha256": _file_digest(self.manifest_path),
            "mask_rate": self.mask_rate,
            "dtype": self.dtype_name,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]


def build_parser(description: str) -> argparse.ArgumentParser:
    """Arguments shared by the mlm and restricted_attention stages."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Run YAML configuration.")
    parser.add_argument("--profile", default=None)
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
        help="Resume without recomputing completed rows (default: true).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse experiment arguments; paths never default relative to the CWD."""

    return build_parser(
        "Score masked-nucleotide accuracy on native transcripts."
    ).parse_args(argv)


def prepare_stage(args: argparse.Namespace, stage: str) -> StageRun:
    """Resolve paths, the runnable registry, the manifest, and runtime settings."""

    project_root = _project_root()
    config_path = _config_path(args.config, project_root)
    experiment = load_experiment_config(config_path, profile=args.profile)
    config = dict(experiment.values)
    config_dir = experiment.config_path.parent

    models_path = _resolve_input_path(
        args.models_path or experiment.models_registry_path,
        project_root=project_root,
        config_dir=config_dir,
    )
    manifest_path = _resolve_input_path(
        args.manifest or experiment.output_manifest_path,
        project_root=project_root,
        config_dir=config_dir,
    )
    output_dir = _resolve_output_path(
        args.output or (project_root / "outputs" / experiment.profile / stage),
        project_root=project_root,
    )
    if not models_path.is_file():
        raise FileNotFoundError(f"models.yaml not found: {models_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_model_registry(
        models_path, config_dir=config_dir, project_root=project_root
    )
    if args.model:
        selected = {str(name) for name in args.model}
        unknown = sorted(selected - registry.keys())
        if unknown:
            raise ValueError(f"Unknown model filters: {unknown}")
        registry = {name: spec for name, spec in registry.items() if name in selected}
    skipped_models: list[dict[str, str]] = []
    runnable: dict[str, ModelSpec] = {}
    for name, spec in registry.items():
        available, reason = backend_runtime_available(spec.backend)
        if available:
            available, reason = weights_available(spec)
        if not available:
            skipped_models.append(
                {"model": name, "backend": spec.backend or "rnabert", "reason": reason}
            )
            print(f"skip {name} ({spec.backend}): {reason}", flush=True)
            continue
        runnable[name] = spec

    manifest_jobs = load_manifest(manifest_path)
    requested_max_length = max(int(job["length"]) for job in manifest_jobs)
    if args.max_jobs is not None:
        if args.max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        manifest_jobs = manifest_jobs[: args.max_jobs]

    cosine_config = config["cosine"]
    dtype_name = args.dtype or str(cosine_config.get("dtype", "bfloat16"))
    resume = (
        args.resume
        if args.resume is not None
        else _as_bool(cosine_config.get("resume", True), "cosine.resume")
    )
    return StageRun(
        project_root=project_root,
        config=config,
        config_path=config_path,
        models_path=models_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        requested_models=list(registry),
        runnable=runnable,
        skipped_models=skipped_models,
        manifest_jobs=manifest_jobs,
        requested_max_length=requested_max_length,
        mask_rate=float(config["mlm"]["mask_rate"]),
        dtype_name=dtype_name,
        dtype=dtype_from_name(dtype_name),
        device=resolve_device(args.device or str(cosine_config.get("device", "auto"))),
        resume=resume,
    )


def start_metadata(stage_run: StageRun, stage: str) -> dict[str, Any]:
    """Validate the resume fingerprint and return the running metadata record."""

    fingerprint = stage_run.fingerprint()
    path = stage_run.output_dir / "run_metadata.json"
    previous: dict[str, Any] = {}
    if stage_run.resume and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        existing = previous.get("run_fingerprint")
        if existing is not None and str(existing) != fingerprint:
            raise ValueError(
                f"Refusing to resume {stage} output with a different run "
                f"fingerprint: existing={existing}, requested={fingerprint}. "
                "Use a new --output directory or --no-resume."
            )
    selected = list(previous.get("selected_models") or [])
    selected += [name for name in stage_run.requested_models if name not in selected]
    metadata = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": fingerprint,
        "selected_models": selected,
        **stage_run.metadata(),
    }
    write_metadata(path, metadata)
    return metadata


def write_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def masked_positions(job: Mapping[str, Any], mask_rate: float) -> list[int]:
    """Return the model-independent masked positions of one transcript."""

    length = int(job["length"])
    count = max(1, int(round(mask_rate * length)))
    rng = random.Random(_job_metric_seed(job, "mlm"))
    return sorted(rng.sample(range(length), count))


def _autocast(loaded: Any) -> Any:
    if loaded.dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=loaded.device.type, dtype=loaded.dtype)


def _nucleotide_logits(
    loaded: Any, sequence: str, positions: Sequence[int]
) -> torch.Tensor:
    """Return logits over A, C, G, T at the masked positions, shape ``(n, 4)``."""

    backend = (loaded.spec.backend or "rnabert").lower()
    normalized = normalize_sequence(sequence)
    if backend in HF_DIR_BACKENDS:
        tokenizer = loaded.tokenizer
        configure_runtime_rope_scaling(
            loaded.model,
            real_sequence_length=len(normalized),
            policy=loaded.rope_scaling,
        )
        token_ids = [tokenizer.token_to_id(base) for base in normalized]
        for position in positions:
            token_ids[position] = tokenizer.mask_token_id
        input_ids = torch.tensor(
            [[tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id]],
            dtype=torch.long,
            device=loaded.device,
        )
        with torch.inference_mode(), _autocast(loaded):
            logits = loaded.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                return_dict=True,
            ).logits
        columns = [tokenizer.token_to_id(base) for base in NUCLEOTIDES]
        offset = 1
    elif backend in {"rnaverse", "ribogram"}:
        if backend == "rnaverse":
            from .rnaverse_io import RNAverseScorer as Scorer
        else:
            from .ribogram_io import RiboGRAMScorer as Scorer

        scorer = Scorer(loaded)
        tokens = scorer.encode(normalized).clone()
        offset = int(scorer.offset)
        masked = torch.tensor(list(positions), dtype=torch.long, device=tokens.device)
        tokens[0, masked + offset] = scorer.mask_id
        logits = scorer.log_probs(tokens)
        columns = [scorer.nt_id(base) for base in NUCLEOTIDES]
    elif backend == "hydrarna":
        from .hydrarna_io import encode_hydrarna_sequence

        dictionary = loaded.dictionary
        mask_id = int(dictionary.index("<mask>"))
        columns = [int(dictionary.index(base)) for base in NUCLEOTIDES]
        if len({mask_id, int(dictionary.unk()), *columns}) != len(NUCLEOTIDES) + 2:
            raise RuntimeError("HydraRNA dictionary lacks <mask> or a nucleotide")
        tokens = encode_hydrarna_sequence(normalized, dictionary, loaded.device).clone()
        offset = 1
        masked = torch.tensor(list(positions), dtype=torch.long, device=tokens.device)
        tokens[0, masked + offset] = mask_id
        with torch.inference_mode(), _autocast(loaded):
            outputs = loaded.model.encoder(tokens)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    else:
        raise ValueError(f"No masked-nucleotide scoring for backend {backend!r}")
    index = torch.tensor(list(positions), dtype=torch.long, device=logits.device)
    picked = logits[0].index_select(0, index + offset)[:, columns].float()
    if picked.shape != (len(positions), len(NUCLEOTIDES)):
        raise RuntimeError(f"Unexpected masked logits shape {tuple(picked.shape)}")
    return picked


def masked_accuracy(loaded: Any, sequence: str, positions: Sequence[int]) -> float:
    """Fraction of masked positions whose top nucleotide is the original one.

    Every selected position becomes the mask token. The prediction is the
    argmax over the four nucleotide logits, so vocabularies of different
    sizes are scored alike.
    """

    normalized = normalize_sequence(sequence)
    unexpected = sorted(
        {normalized[position] for position in positions} - set(NUCLEOTIDES)
    )
    if unexpected:
        raise ValueError(f"Masked positions hold non-ACGT symbols {unexpected}")
    target = torch.tensor(
        [NUCLEOTIDES.index(normalized[position]) for position in positions],
        dtype=torch.long,
    )
    predicted = _nucleotide_logits(loaded, normalized, positions).argmax(dim=-1).cpu()
    if loaded.device.type == "cuda":
        torch.cuda.empty_cache()
    return int((predicted == target).sum().item()) / len(positions)


def run(args: argparse.Namespace) -> None:
    """Score every selected model, loading one model at a time."""

    stage_run = prepare_stage(args, "mlm")
    native_jobs = [
        job for job in stage_run.manifest_jobs if job["control_type"] == "native"
    ]
    if not native_jobs:
        raise ValueError("No native transcripts remain after filtering")
    metadata = start_metadata(stage_run, "mlm")
    metadata["selected_jobs"] = [job["job_id"] for job in native_jobs]
    metadata["definitions"] = {
        **MLM_DEFINITIONS,
        "statistical_unit": "native transcript",
    }
    metadata_path = stage_run.output_dir / "run_metadata.json"
    write_metadata(metadata_path, metadata)

    sink = CsvSink(
        stage_run.output_dir / "mlm_units.csv",
        MLM_FIELDS,
        ("model", "job_id"),
        resume=stage_run.resume,
    )
    try:
        for model_name, spec in stage_run.runnable.items():
            pending = [
                job
                for job in native_jobs
                if (model_name, job["job_id"]) not in sink.completed
            ]
            if not pending:
                continue
            loaded = load_encoder(
                spec,
                device=stage_run.device,
                dtype=stage_run.dtype,
                requested_max_length=stage_run.requested_max_length,
            )
            try:
                for job in pending:
                    positions = masked_positions(job, stage_run.mask_rate)
                    accuracy = masked_accuracy(loaded, str(job["sequence"]), positions)
                    sink.append(
                        [
                            {
                                "model": model_name,
                                "model_path": str(spec.path),
                                "job_id": job["job_id"],
                                "pair_group_id": job["pair_group_id"],
                                "seed": job["seed"],
                                "length": job["length"],
                                "length_group": job["length_group"],
                                "mask_rate": stage_run.mask_rate,
                                "masked_positions": len(positions),
                                "mlm_accuracy": accuracy,
                            }
                        ]
                    )
            finally:
                loaded.close()
                del loaded
                gc.collect()
            print(f"mlm {model_name}: scored {len(pending)} transcripts", flush=True)
    finally:
        sink.close()

    metadata["status"] = "completed"
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_metadata(metadata_path, metadata)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    run(parse_args(argv))


if __name__ == "__main__":
    main()
