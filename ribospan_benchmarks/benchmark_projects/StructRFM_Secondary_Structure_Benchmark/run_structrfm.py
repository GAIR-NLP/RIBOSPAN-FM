#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Frozen-backbone secondary-structure runner for the structRFM protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import registry_rows  # noqa: E402
from scripts.official import existing_table, score_published  # noqa: E402
from scripts.stages import embed_task, score_task, train_task  # noqa: E402
from scripts.tasks import TASKS  # noqa: E402

STAGES = ("check", "embed", "train", "score", "summarize", "all")


def run_check() -> int:
    print("model\tbackend\tcontext\tcheckpoint\tstatus\tbytes", flush=True)
    missing = False
    for row in registry_rows():
        for relative, complete, nbytes in row["checkpoints"]:
            if not complete:
                missing = True
            status = "ready" if complete else "partial"
            print(
                f"{row['name']}\t{row['backend']}\t{row['context_window']}\t{relative}\t{status}\t{nbytes}",
                flush=True,
            )
    return 1 if missing else 0


def run_summarize() -> None:
    from scripts.summarize import summarize

    paths = summarize()
    for path in paths:
        print(f"wrote {path.relative_to(ROOT)}", flush=True)
    print(paths[0].read_text(encoding="utf-8"), flush=True)


def _ribospan_models(args: argparse.Namespace) -> list[str]:
    rows = [row for row in registry_rows() if row["backend"] == "ribospan"]
    names = {row["name"] for row in rows}
    if args.model is None:
        return [row["name"] for row in rows]
    if args.model == "structRFM":
        return []
    if args.model not in names:
        raise SystemExit(f"unknown model {args.model}")
    return [args.model]


def _tasks(args: argparse.Namespace) -> list[str]:
    return [args.task] if args.task else list(TASKS)


def _score_official(args: argparse.Namespace, device: torch.device) -> None:
    if args.stage not in {"all", "score"}:
        return
    if args.model not in {None, "structRFM"}:
        return
    for task in _tasks(args):
        found = existing_table(task)
        if found is not None and not args.no_skip_existing:
            print(f"skip existing structRFM/{task} {found}", flush=True)
            continue
        score_published(task, device, args.limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES)
    parser.add_argument("--model", default=None, help="structRFM or one RIBOSPAN checkpoint. Default: all.")
    parser.add_argument("--task", default=None, choices=tuple(TASKS), help="One task. Default: both.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    if args.stage == "check":
        return run_check()
    if args.stage == "summarize":
        run_summarize()
        return 0
    if args.model not in {None, "structRFM"} and args.model not in {row["name"] for row in registry_rows()}:
        raise SystemExit(f"unknown model {args.model}")
    device = torch.device(args.device)
    if args.model == "structRFM" and args.stage in {"embed", "train"}:
        print(f"skip structRFM {args.stage}", flush=True)
        return 0
    models = _ribospan_models(args)
    tasks = _tasks(args)
    stages = ("embed", "train") if args.stage == "all" else (args.stage,)
    root = ROOT / "outputs" / "ssp_structrfm" / "heads"
    for model in models:
        for task in tasks:
            done = any((root / model / task).glob("test_metric_*.csv"))
            if done and not args.no_skip_existing and args.stage in {"all", "train", "score"}:
                print(f"skip existing {model}/{task}", flush=True)
                continue
            if "embed" in stages:
                embed_task(model, task, device, args.limit)
            if "score" in stages:
                score_task(model, task, device, epoch=args.epoch, limit=args.limit)
            if "train" in stages:
                train_task(model, task, device, epochs=args.epochs, lr=args.lr, limit=args.limit)
    _score_official(args, device)
    if args.stage == "all":
        run_summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
