# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Embed frozen RIBOSPAN states, then train and score the structRFM head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from scripts.config import PROJECT_ROOT, registry_rows
from scripts.head_train import (
    build_head,
    load_checkpoint,
    predict_one,
    save_checkpoint,
    train_one,
    write_json,
)
from scripts.ribospan_features import embed_cls_nucleotides, load_frozen_ribospan
from scripts.tasks import SSP_ROOT, TASKS, list_bpseq, prepare_example

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ssp_structrfm"


def _split_paths(task: str, split: str) -> list[Path]:
    spec = TASKS[task]
    directory = SSP_ROOT / spec[split]
    max_length = spec["test_max_length"] if split == "test" else None
    paths = list_bpseq(directory, max_length=max_length)
    if not paths:
        raise FileNotFoundError(f"no bpseq under {directory}")
    return paths


def _manifest_path(model: str, task: str, split: str) -> Path:
    return OUTPUT_ROOT / "embeddings" / model / task / f"{split}.jsonl"


def _embedding_path(model: str, task: str, split: str, index: int) -> Path:
    return OUTPUT_ROOT / "embeddings" / model / task / split / f"{index:06d}.npy"


def embed_task(model: str, task: str, device: torch.device, limit: int | None) -> None:
    row = next(item for item in registry_rows() if item["name"] == model)
    if row["backend"] != "ribospan":
        raise ValueError(f"{model} is not a RIBOSPAN checkpoint")
    loaded = load_frozen_ribospan(model, row["path"], device)
    for split in ("train", "test"):
        paths = _split_paths(task, split)
        if limit is not None:
            paths = paths[:limit]
        manifest = _manifest_path(model, task, split)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for index, source in enumerate(paths):
            dest = _embedding_path(model, task, split, index)
            seq, _pairs = prepare_example(source)
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                array = embed_cls_nucleotides(loaded, seq)
                np.save(dest, array)
            lines.append(
                json.dumps(
                    {
                        "index": index,
                        "bpseq": str(source),
                        "embedding": str(dest),
                        "n_nt": len(seq),
                    }
                )
            )
            if index % 50 == 0:
                print(f"embed {model} {task} {split} {index + 1}/{len(paths)}", flush=True)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {manifest}", flush=True)
    model, _tokenizer = loaded
    model.to("cpu")
    del model
    torch.cuda.empty_cache()


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _latest_checkpoint(directory: Path) -> Path | None:
    paths = sorted(directory.glob("model_ep*.pt"))
    return paths[-1] if paths else None


TEST_METRIC_COLUMNS = ("name", "f1s", "accuracy", "precision", "recall")


def _sequence_name(bpseq: str) -> str:
    """Match structRFM: basename, and drop a trailing bpseq/ct/dbn suffix."""
    name = Path(bpseq).name
    if any(name.endswith(suffix) for suffix in (".bpseq", ".ct", ".dbn")):
        name = name[: name.rfind(".")]
    return name


def _score_records(model, rows: list[dict], device: torch.device, label: str) -> list[dict]:
    records = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        seq, pairs = prepare_example(Path(row["bpseq"]))
        metrics = predict_one(model, seq, pairs, Path(row["embedding"]), device)
        records.append({"name": _sequence_name(row["bpseq"]), **metrics})
        if index % 200 == 0 or index == total:
            print(f"score {label} {index}/{total}", flush=True)
    return records


def _mean_metrics(records: list[dict]) -> dict[str, float]:
    keys = [key for key in TEST_METRIC_COLUMNS if key != "name"]
    return {key: float(np.mean([item[key] for item in records])) for key in keys}


def _write_test_metric(path: Path, records: list[dict]) -> None:
    """Write structRFM's per-sequence table: name,f1s,accuracy,precision,recall."""
    import pandas as pd

    frame = pd.DataFrame(records, columns=list(TEST_METRIC_COLUMNS))
    frame.to_csv(path, index=False)


def _history_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _best_f1_epoch(history: list[dict]) -> int:
    return int(max(history, key=lambda row: row["f1s"])["epoch"])


def _metric_filename(epoch: int, f1: float) -> str:
    return f"test_metric_{epoch}_{f1:.4f}.csv"


def _keep_best_test_metric(directory: Path, epoch: int, f1: float, records: list[dict]) -> Path:
    """Leave one table, the epoch whose mean F1 is the best so far."""
    path = directory / _metric_filename(epoch, f1)
    for old in directory.glob("test_metric_*.csv"):
        old.unlink()
    _write_test_metric(path, records)
    return path


def train_task(
    model_name: str,
    task: str,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    limit: int | None,
) -> None:
    train_rows = _read_manifest(_manifest_path(model_name, task, "train"))
    test_rows = _read_manifest(_manifest_path(model_name, task, "test"))
    if limit is not None:
        train_rows = train_rows[:limit]
        test_rows = test_rows[:limit]
    sample = np.load(train_rows[0]["embedding"])
    embed_dim = int(sample.shape[1])
    model, loss_fn = build_head(embed_dim, device)
    out_dir = OUTPUT_ROOT / "heads" / model_name / task
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"
    start_epoch = 0
    latest = _latest_checkpoint(out_dir)
    if latest is not None:
        start_epoch = load_checkpoint(latest, model)
        print(f"resume {latest} next_epoch={start_epoch}", flush=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    history = [row for row in _history_rows(history_path) if row["epoch"] < start_epoch]
    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0.0
        seen = 0
        for row in train_rows:
            seq, pairs = prepare_example(Path(row["bpseq"]))
            value = train_one(
                model,
                loss_fn,
                optimizer,
                seq,
                pairs,
                Path(row["embedding"]),
                device,
            )
            total += value
            seen += 1
            if seen % 50 == 0:
                print(
                    f"train {model_name} {task} epoch {epoch} {seen}/{len(train_rows)} "
                    f"loss={total / seen:.4f}",
                    flush=True,
                )
        model.eval()
        records = _score_records(model, test_rows, device, f"{model_name} {task} epoch {epoch}")
        metrics = _mean_metrics(records)
        record = {"epoch": epoch, "train_loss": total / max(seen, 1), **metrics}
        previous_best = max((row["f1s"] for row in history), default=0.0)
        history.append(record)
        save_checkpoint(out_dir / f"model_ep{epoch:03d}.pt", model, epoch, metrics["f1s"])
        write_json(history_path, history)
        print(
            f"Test\t{task}\tf1s: {metrics['f1s']:.4f}",
            flush=True,
        )
        if metrics["f1s"] > previous_best:
            csv_path = _keep_best_test_metric(out_dir, epoch, metrics["f1s"], records)
            print(f"wrote {csv_path}", flush=True)
        else:
            print(
                f"kept test table: epoch {epoch} F1 {metrics['f1s']:.4f} "
                f"is below {previous_best:.4f}",
                flush=True,
            )
        print(
            f"epoch {epoch} loss={record['train_loss']:.4f} test_F1={metrics['f1s']:.4f}",
            flush=True,
        )


def score_task(
    model_name: str,
    task: str,
    device: torch.device,
    *,
    epoch: int | None,
    limit: int | None,
) -> None:
    """Score one saved head and write structRFM's per-sequence test table."""
    test_rows = _read_manifest(_manifest_path(model_name, task, "test"))
    if limit is not None:
        test_rows = test_rows[:limit]
    out_dir = OUTPUT_ROOT / "heads" / model_name / task
    history = _history_rows(out_dir / "history.json")
    if epoch is None:
        if not history:
            raise FileNotFoundError(f"no history in {out_dir}; pass --epoch")
        epoch = _best_f1_epoch(history)
    checkpoint = out_dir / f"model_ep{epoch:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sample = np.load(test_rows[0]["embedding"])
    model, _loss_fn = build_head(int(sample.shape[1]), device)
    load_checkpoint(checkpoint, model)
    model.eval()
    records = _score_records(model, test_rows, device, f"{model_name} {task} epoch {epoch}")
    metrics = _mean_metrics(records)
    print(f"Test\t{task}\tf1s: {metrics['f1s']:.4f}", flush=True)
    if history and epoch != _best_f1_epoch(history):
        best_epoch = _best_f1_epoch(history)
        print(
            f"kept test table: epoch {epoch} is below best epoch {best_epoch}",
            flush=True,
        )
        return
    csv_path = _keep_best_test_metric(out_dir, epoch, metrics["f1s"], records)
    print(f"wrote {csv_path}", flush=True)
