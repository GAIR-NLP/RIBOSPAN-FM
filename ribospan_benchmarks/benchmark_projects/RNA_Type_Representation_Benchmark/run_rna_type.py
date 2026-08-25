#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stage runner: embed -> rfam -> analyze -> plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import load_experiment_config, resolve_config_path  # noqa: E402

STAGES = ("embed", "rfam", "analyze", "plot")


def _call(label: str, func, argv: list[str]) -> None:
    print("+", label, " ".join(argv), flush=True)
    rc = func(argv)
    if rc not in (None, 0):
        raise SystemExit(rc)


def _stage_argv(args: argparse.Namespace, extra: list[str] | None = None) -> list[str]:
    argv = ["--config", str(resolve_config_path(args.config))]
    if args.models:
        argv += ["--models", *args.models]
    if extra:
        argv += extra
    return argv


def run_embed(args: argparse.Namespace) -> None:
    from scripts.extract import main as embed_main

    extra: list[str] = []
    if args.device:
        extra += ["--device", args.device]
    _call("embed", embed_main, _stage_argv(args, extra))


def run_rfam(args: argparse.Namespace) -> None:
    from scripts.rfam import main as rfam_main

    _call("rfam", rfam_main, ["--config", str(resolve_config_path(args.config))])


def run_analyze(args: argparse.Namespace) -> None:
    from scripts.analyze import main as analyze_main

    extra = ["--top-k-families", "20"]
    _call("analyze", analyze_main, _stage_argv(args, extra))


def run_plot(args: argparse.Namespace) -> None:
    from scripts.plotting.atlas import main as atlas_main
    from scripts.plotting.composite import main as composite_main
    from scripts.plotting.long import main as long_main
    from scripts.plotting.rfam import main as rfam_plot_main

    recompute = ["--recompute-tsne"] if args.recompute_tsne else []
    _call(
        "plot.rfam",
        rfam_plot_main,
        _stage_argv(args, ["--top-k-families", "20", *recompute]),
    )
    _call("plot.long", long_main, _stage_argv(args, recompute))
    _call(
        "plot.atlas",
        atlas_main,
        _stage_argv(
            args,
            [
                "--pooling",
                "mean",
                "--max-points",
                "0",
                "--no-pca",
                "--no-scaler",
                *recompute,
            ],
        ),
    )

    cfg, _ = load_experiment_config(resolve_config_path(args.config))
    canonical = list(cfg["models"]["names"])
    if args.models and list(args.models) != canonical:
        print(
            "skipping canonical composite (partial model list); "
            "pass scripts.plotting.composite --out explicitly if needed",
            flush=True,
        )
        return
    _call("plot.composite", composite_main, _stage_argv(args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=(*STAGES, "all"),
        help="Pipeline stage; default all.",
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--recompute-tsne",
        action="store_true",
        help="Recompute t-SNE instead of reusing saved coordinates.",
    )
    args = parser.parse_args()

    requested = list(STAGES if args.stage == "all" else (args.stage,))
    runners = {
        "embed": run_embed,
        "rfam": run_rfam,
        "analyze": run_analyze,
        "plot": run_plot,
    }
    for stage in requested:
        runners[stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
