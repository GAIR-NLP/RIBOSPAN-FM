#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stage runner: score -> summarize."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import resolve_config_path  # noqa: E402

STAGES = ("score", "summarize")


def _call(label: str, func, argv: list[str]) -> None:
    print("+", label, " ".join(argv), flush=True)
    rc = func(argv)
    if rc not in (None, 0):
        raise SystemExit(rc)


def _shared_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--config", str(resolve_config_path(args.config))]
    if args.group:
        argv += ["--group", args.group]
    if args.datasets:
        argv += ["--datasets", *args.datasets]
    return argv


def run_score(args: argparse.Namespace) -> None:
    from scripts.score import main as score_main

    extra: list[str] = []
    if args.models:
        extra += ["--models", *args.models]
    if args.device:
        extra += ["--device", args.device]
    if args.no_skip_existing:
        extra.append("--no-skip-existing")
    _call("score", score_main, _shared_argv(args) + extra)


def run_summarize(args: argparse.Namespace) -> None:
    from scripts.summarize import main as summarize_main

    _call("summarize", summarize_main, ["--config", str(resolve_config_path(args.config))])


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
    parser.add_argument("--group", default=None, help="all | smoke | ribozyme | trna | aptamer | mrna-coding | mrna-splicing")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    requested = list(STAGES if args.stage == "all" else (args.stage,))
    runners = {
        "score": run_score,
        "summarize": run_summarize,
    }
    for stage in requested:
        runners[stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
