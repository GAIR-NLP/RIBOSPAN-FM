#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stage runner for the LIMBER/RiNALMo secondary-structure protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import load_experiment_config, model_rows, output_dir, task_spec  # noqa: E402
from scripts.decoder import prob_mat_to_sec_struct  # noqa: E402
from scripts.metrics import assert_protocol_cases  # noqa: E402

STAGES = ("check", "train", "summarize", "all")
ARCHIVE_FAMILIES = (
    "16s",
    "23s",
    "5s",
    "RNaseP",
    "grp1",
    "srp",
    "tRNA",
    "telomerase",
    "tmRNA",
)


def _exists(path: Path | None) -> str:
    if path is None:
        return "unset"
    return "ready" if path.exists() else "missing"


def run_check(config_path: str | None) -> int:
    assert_protocol_cases()
    import numpy as np

    # G-C pairs at (0, 4) and (0, 5). The higher probability keeps the base.
    probabilities = np.zeros((6, 6), dtype=np.float64)
    probabilities[0, 5] = probabilities[5, 0] = 0.9
    probabilities[0, 4] = probabilities[4, 0] = 0.8
    decoded = prob_mat_to_sec_struct(probabilities, "GAAACC", threshold=0.5)
    if decoded[0, 5] != 1 or decoded[0, 4] != 0:
        raise AssertionError("greedy decoder kept the lower-probability competing pair")
    config, root = load_experiment_config(config_path)
    print("metric\tf1_shift\tdecoder\tstatus", flush=True)
    print("flexible_f1\t1.0000\tgreedy\tok", flush=True)
    print("model\tbackend\tweights\tstatus", flush=True)
    missing = False
    for row in model_rows(config, root):
        weights = Path(row["weights"])
        if row["backend"] == "limber":
            ready = (weights / "pytorch_model.bin").is_file() and (weights / "model_config.json").is_file()
        else:
            ready = (weights / "config.json").is_file() and (weights / "pytorch_model.bin").is_file()
        if not ready:
            missing = True
        print(
            f"{row['name']}\t{row['backend']}\t{weights}\t{'ready' if ready else 'missing'}",
            flush=True,
        )
    print("task\tsplit\tpath\tstatus", flush=True)
    for name in config.get("tasks", {}):
        spec = task_spec(config, root, name)
        for key in ("train", "valid", "test", "root"):
            if key not in spec:
                continue
            path = spec[key]
            status = _exists(path)
            if status == "missing":
                missing = True
            print(f"{name}\t{key}\t{path}\t{status}", flush=True)
    return 1 if missing else 0


def run_summarize(config: dict, root: Path) -> None:
    from scripts.summarize import summarize

    paths = summarize(config, root, ARCHIVE_FAMILIES)
    for path in paths:
        print(f"wrote {path.relative_to(root)}", flush=True)
    print(paths[0].read_text(encoding="utf-8"), flush=True)


def _jobs(config: dict, root: Path, args: argparse.Namespace) -> list[tuple[str, str, str | None]]:
    names = [args.model] if args.model else [row["name"] for row in model_rows(config, root)]
    tasks = [args.task] if args.task else list(config.get("tasks", {}))
    jobs: list[tuple[str, str, str | None]] = []
    for name in names:
        for task in tasks:
            if task == "archiveII_lfo":
                families = (args.family,) if args.family else ARCHIVE_FAMILIES
                jobs.extend((name, task, family) for family in families)
            else:
                jobs.append((name, task, None))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", default=None, help="One registry model. Default: every model.")
    parser.add_argument(
        "--task",
        default=None,
        choices=("bprna_ts0", "rivas_testsetb", "archiveII_lfo"),
        help="One task. Default: every task.",
    )
    parser.add_argument("--family", default=None, help="One ArchiveII family. Default: all nine.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train", type=int, default=0, help="Cap the training split. 0 uses all of it.")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    if args.stage == "check":
        return run_check(args.config)
    config, root = load_experiment_config(args.config)
    if args.stage == "summarize":
        run_summarize(config, root)
        return 0
    from scripts.train import train_protocol

    for model_name, task_name, family in _jobs(config, root, args):
        destination = output_dir(config, root, model_name, task_name, family)
        if not args.no_skip_existing and (destination / "result.json").is_file():
            print(f"skip existing {destination.relative_to(root)}", flush=True)
            continue
        train_protocol(
            config,
            root,
            model_name=model_name,
            task_name=task_name,
            device=torch.device(args.device),
            family=family,
            max_train=args.max_train,
        )
    if args.stage == "all":
        run_summarize(config, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
