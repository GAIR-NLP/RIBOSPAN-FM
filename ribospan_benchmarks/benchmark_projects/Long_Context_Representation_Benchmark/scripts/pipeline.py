#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run the shared RNA contextualization manifest, cosine, and attention stages."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import load_experiment_config, public_path
from .generation import generate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("generate", "attention", "cosine", "analyze")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(*STAGES, "all"),
        help="Pipeline stage to run; all executes stages in order.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default=None
    )
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--attention-layers", nargs="+", default=None)
    parser.add_argument("--attention-heads", nargs="+", default=None)
    parser.add_argument("--heatmap-jobs", nargs="*", default=None)
    parser.add_argument("--heatmap-window", type=int, default=None)
    parser.add_argument("--heatmap-pool-size", type=int, default=None)
    parser.add_argument("--heatmap-norm-quantile", type=float, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args(argv)


def _paths(args: argparse.Namespace) -> tuple[Any, Path, Path]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    experiment = load_experiment_config(config_path, profile=args.profile)
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else PROJECT_ROOT / "outputs" / experiment.profile
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    manifest = (
        Path(args.manifest).expanduser()
        if args.manifest
        else output_root / "manifest.jsonl"
    )
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    return experiment, output_root, manifest.resolve()


def _run_generate(
    args: argparse.Namespace, experiment: Any, manifest_path: Path
) -> dict[str, Any]:
    return generate_manifest(
        experiment,
        output_path=manifest_path,
        max_groups=args.max_groups,
    )


def _run_cosine(
    args: argparse.Namespace,
    experiment: Any,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    from . import cosine

    argv = [
        "--config",
        str(experiment.config_path),
        "--profile",
        experiment.profile,
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_root / "cosine"),
    ]
    if args.registry:
        argv.extend(["--models", args.registry])
    for model in args.models or ():
        argv.extend(["--model", model])
    if args.device:
        argv.extend(["--device", args.device])
    if args.dtype:
        argv.extend(["--dtype", args.dtype])
    if args.max_jobs is not None:
        argv.extend(["--max-jobs", str(args.max_jobs)])
    if args.no_resume:
        argv.append("--no-resume")
    cosine.main(argv)
    metadata_path = output_root / "cosine" / "run_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _run_attention(
    args: argparse.Namespace,
    experiment: Any,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    from . import attention

    argv = [
        "--config",
        str(experiment.config_path),
        "--profile",
        experiment.profile,
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_root / "attention"),
    ]
    if args.registry:
        argv.extend(["--registry", args.registry])
    if args.models:
        argv.extend(["--models", *args.models])
    if args.device:
        argv.extend(["--device", args.device])
    if args.dtype:
        argv.extend(["--dtype", args.dtype])
    if args.max_jobs is not None:
        argv.extend(["--max-jobs", str(args.max_jobs)])
    if args.summary_only:
        argv.append("--summary-only")
    if args.attention_layers:
        argv.extend(["--layers", *args.attention_layers])
    if args.attention_heads:
        argv.extend(["--heads", *args.attention_heads])
    if args.heatmap_jobs is not None:
        argv.extend(["--heatmap-jobs", *args.heatmap_jobs])
    if args.heatmap_window is not None:
        argv.extend(["--heatmap-window", str(args.heatmap_window)])
    if args.heatmap_pool_size is not None:
        argv.extend(["--heatmap-pool-size", str(args.heatmap_pool_size)])
    if args.heatmap_norm_quantile is not None:
        argv.extend(
            ["--heatmap-norm-quantile", str(args.heatmap_norm_quantile)]
        )
    parsed = attention.parse_args(argv)
    return attention.run(parsed)


def _run_analyze(
    args: argparse.Namespace,
    experiment: Any,
    output_root: Path,
) -> dict[str, Any]:
    from . import analysis

    argv = [
        "--config",
        str(experiment.config_path),
        "--profile",
        experiment.profile,
        "--cosine-dir",
        str(output_root / "cosine"),
        "--attention-dir",
        str(output_root / "attention"),
        "--output",
        str(output_root / "analysis"),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
    ]
    return analysis.run(analysis.parse_args(argv))


def run(args: argparse.Namespace) -> dict[str, Any]:
    experiment, output_root, manifest_path = _paths(args)
    requested = list(STAGES if args.stage == "all" else (args.stage,))
    if args.skip_generate:
        requested = [stage for stage in requested if stage != "generate"]
    if args.skip_attention:
        requested = [stage for stage in requested if stage != "attention"]
    if args.skip_analyze:
        requested = [stage for stage in requested if stage != "analyze"]
    if (
        "generate" not in requested
        and any(stage != "generate" for stage in requested)
        and not manifest_path.exists()
    ):
        requested.insert(0, "generate")

    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "config": public_path(experiment.config_path, PROJECT_ROOT),
        "profile": experiment.profile,
        "manifest": public_path(manifest_path, PROJECT_ROOT),
        "output_root": public_path(output_root, PROJECT_ROOT),
        "requested_stages": requested,
        "stages": {},
    }
    state_path = output_root / "pipeline_state.json"
    _write_json(state_path, state)

    runners = {
        "generate": lambda: _run_generate(args, experiment, manifest_path),
        "cosine": lambda: _run_cosine(
            args, experiment, output_root, manifest_path
        ),
        "attention": lambda: _run_attention(
            args, experiment, output_root, manifest_path
        ),
        "analyze": lambda: _run_analyze(args, experiment, output_root),
    }
    try:
        for stage in requested:
            state["stages"][stage] = {
                "status": "running",
                "started_at": _utc_now(),
            }
            _write_json(state_path, state)
            result = runners[stage]()
            state["stages"][stage] = {
                "status": "completed",
                "finished_at": _utc_now(),
                "result": result,
            }
            _write_json(state_path, state)
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        raise
    finally:
        state["finished_at"] = _utc_now()
        _write_json(state_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    state = run(_parse_args(argv))
    print(
        json.dumps(
            {
                "status": state["status"],
                "profile": state["profile"],
                "output_root": state["output_root"],
                "stages": list(state["stages"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

