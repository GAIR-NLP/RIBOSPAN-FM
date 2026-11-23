# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Gradual-unfreeze secondary-structure fine-tuning from the LIMBER/RiNALMo script."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.backbones import StructureBackbone, load_backbone
from scripts.config import model_rows, output_dir, task_spec
from scripts.data import load_ss_dir
from scripts.decoder import prob_mat_to_sec_struct
from scripts.head import SecStructPredictionHead
from scripts.metrics import THRESHOLDS, ss_f1, ss_precision, ss_recall

LR_DECAY_STEPS = 7000


def _split_or_holdout(
    spec: dict,
    max_len: int,
    seed: int,
    max_train: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if spec.get("root"):
        family = spec.get("family")
        if not family:
            raise ValueError("an ArchiveII leave-one-family run needs --family")
        root = Path(spec["root"]) / family
        train, _ = load_ss_dir(root / "train", max_len)
        valid, _ = load_ss_dir(root / "valid", max_len)
        test, _ = load_ss_dir(root / "test", max_len)
    else:
        train_dir = spec.get("train")
        test_dir = spec.get("test")
        if train_dir is None or test_dir is None:
            raise FileNotFoundError(f"task is missing train or test: {spec}")
        if not Path(train_dir).is_dir() or not Path(test_dir).is_dir():
            raise FileNotFoundError(f"missing split directory under {train_dir} or {test_dir}")
        train, skipped_train = load_ss_dir(Path(train_dir), max_len)
        test, skipped_test = load_ss_dir(Path(test_dir), max_len)
        valid_dir = spec.get("valid")
        if valid_dir is not None and Path(valid_dir).is_dir():
            valid, skipped_valid = load_ss_dir(Path(valid_dir), max_len)
        else:
            if valid_dir is not None:
                raise FileNotFoundError(
                    f"validation split is not on disk: {valid_dir}. "
                    "bpRNA VL0 is not in the structRFM release; place the SPOT-RNA VL0 "
                    "files there before training TS0."
                )
            fraction = float(spec.get("val_frac", 0.10))
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(train))
            holdout = max(1, int(len(train) * fraction))
            valid = [train[index] for index in order[:holdout]]
            train = [train[index] for index in order[holdout:]]
            skipped_valid = 0
        print(
            f"skipped>{max_len} train/valid/test "
            f"{skipped_train}/{skipped_valid}/{skipped_test}",
            flush=True,
        )
    if max_train:
        train = train[:max_train]
    if not train or not valid or not test:
        raise RuntimeError(
            f"empty split train={len(train)} valid={len(valid)} test={len(test)}"
        )
    return train, valid, test


def _logits(backbone: StructureBackbone, head: SecStructPredictionHead, sequence: str, device: torch.device):
    states = backbone.hidden(sequence, device)
    return head(states)[0]


@torch.no_grad()
def _tune_threshold(backbone, head, items: list[dict], device: torch.device) -> tuple[float, float]:
    backbone.module.eval()
    head.eval()
    per_threshold = {threshold: [] for threshold in THRESHOLDS}
    for item in items:
        probabilities = torch.sigmoid(
            _logits(backbone, head, item["seq"], device)
        ).float().cpu().numpy()
        for threshold in THRESHOLDS:
            pairing = prob_mat_to_sec_struct(probabilities, item["seq"], threshold=threshold)
            per_threshold[threshold].append(ss_f1(item["pair"], pairing))
    means = {
        threshold: statistics.mean(values) if values else 0.0
        for threshold, values in per_threshold.items()
    }
    best = max(means, key=means.get)
    return best, means[best]


@torch.no_grad()
def _score(backbone, head, items: list[dict], device: torch.device, threshold: float) -> dict:
    backbone.module.eval()
    head.eval()
    precision, recall, f1, names = [], [], [], []
    for item in items:
        probabilities = torch.sigmoid(
            _logits(backbone, head, item["seq"], device)
        ).float().cpu().numpy()
        pairing = prob_mat_to_sec_struct(probabilities, item["seq"], threshold=threshold)
        precision.append(ss_precision(item["pair"], pairing))
        recall.append(ss_recall(item["pair"], pairing))
        f1.append(ss_f1(item["pair"], pairing))
        names.append(item["id"])
    return {
        "mean_F1": statistics.mean(f1) if f1 else 0.0,
        "median_F1": statistics.median(f1) if f1 else 0.0,
        "mean_precision": statistics.mean(precision) if precision else 0.0,
        "mean_recall": statistics.mean(recall) if recall else 0.0,
        "per_seq_f1": [round(value, 6) for value in f1],
        "per_seq_precision": [round(value, 6) for value in precision],
        "per_seq_recall": [round(value, 6) for value in recall],
        "per_seq_ids": names,
        "num_test": len(items),
    }


def _unfreeze_schedule(n_layers: int, every: int, block: int, minimum: int) -> dict[int, list[int]]:
    schedule: dict[int, list[int]] = {}
    top = n_layers
    epoch = every
    while top > minimum:
        lower = max(minimum, top - block)
        schedule[epoch] = list(range(top - 1, lower - 1, -1))
        top = lower
        epoch += every
    return schedule


def _set_train_mode(backbone: StructureBackbone, head: nn.Module, unfrozen: set[int]) -> None:
    backbone.module.eval()
    head.train()
    for index in unfrozen:
        backbone.layers[index].train()


def train_protocol(
    config: dict,
    root: Path,
    *,
    model_name: str,
    task_name: str,
    device: torch.device,
    family: str | None,
    max_train: int,
) -> Path:
    rows = model_rows(config, root)
    row = next(item for item in rows if item["name"] == model_name)
    spec = task_spec(config, root, task_name)
    if family:
        spec["family"] = family
    max_len = int(config.get("max_len", 512))
    epochs = int(config.get("epochs", 26))
    base_lr = float(config.get("base_lr", 5e-4))
    every = int(config.get("unfreeze_every", 3))
    block = int(config.get("unfreeze_block", 3))
    minimum = int(config.get("unfreeze_min_layer", 0))
    tune_every = int(config.get("tune_every", 5))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_items, valid_items, test_items = _split_or_holdout(spec, max_len, seed, max_train)
    print(
        f"{model_name} {task_name} train/valid/test "
        f"{len(train_items)}/{len(valid_items)}/{len(test_items)} max_len={max_len}",
        flush=True,
    )
    backbone = load_backbone(row, device)
    head = SecStructPredictionHead(backbone.hidden_size, num_blocks=2).to(device)
    for parameter in backbone.module.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.Adam([{"params": list(head.parameters())}], lr=base_lr)
    base_lrs = [base_lr]
    schedule = _unfreeze_schedule(len(backbone.layers), every, block, minimum)
    loss_fn = nn.BCEWithLogitsLoss()
    step = 0
    unfrozen: set[int] = set()
    best_f1 = -1.0
    best_threshold = 0.5
    best_head = None
    best_model = None
    best_epoch = -1
    history = []
    destination = output_dir(config, root, model_name, task_name, family)
    destination.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if epoch in schedule:
            fresh = []
            for index in schedule[epoch]:
                unfrozen.add(index)
                for parameter in backbone.layers[index].parameters():
                    parameter.requires_grad_(True)
                    fresh.append(parameter)
            optimizer.add_param_group({"params": fresh, "lr": base_lr / 10.0})
            base_lrs.append(base_lr / 10.0)
            print(
                f"epoch {epoch}: unfroze layers {schedule[epoch]} lr={base_lr / 10:.1e}",
                flush=True,
            )
        _set_train_mode(backbone, head, unfrozen)
        order = np.random.permutation(len(train_items))
        total = 0.0
        started = time.perf_counter()
        for index in order:
            item = train_items[int(index)]
            length = len(item["seq"])
            logits = _logits_train(backbone, head, item["seq"], device)
            upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
            target = torch.from_numpy(item["pair"]).to(device)
            loss = loss_fn(logits[upper], target[upper])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            factor = 1.0 - 0.9 * min(step / LR_DECAY_STEPS, 1.0)
            for group_index, group in enumerate(optimizer.param_groups):
                group["lr"] = base_lrs[group_index] * factor
            total += float(loss.item())
        elapsed = time.perf_counter() - started
        mean_loss = total / len(train_items)
        record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "unfrozen_layers": sorted(unfrozen),
            "seconds": elapsed,
        }
        print(
            f"epoch {epoch} train {len(train_items)} in {elapsed:.0f}s "
            f"loss={mean_loss:.4f} unfrozen={len(unfrozen)}",
            flush=True,
        )
        if (epoch + 1) % tune_every == 0 or epoch == epochs - 1:
            threshold, val_f1 = _tune_threshold(backbone, head, valid_items, device)
            record["val_F1"] = val_f1
            record["threshold"] = threshold
            print(
                f"epoch {epoch + 1} val_F1={val_f1:.4f} thr={threshold}",
                flush=True,
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_epoch = epoch
                best_head = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                best_model = {
                    key: value.detach().cpu().clone()
                    for key, value in backbone.module.state_dict().items()
                }
                torch.save(
                    {
                        "model": best_model,
                        "head": best_head,
                        "epoch": epoch,
                        "threshold": threshold,
                        "val_F1": val_f1,
                        "model_name": model_name,
                        "task": task_name,
                    },
                    destination / "best.pt",
                )
        history.append(record)
        (destination / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if best_head is None or best_model is None:
        raise RuntimeError("validation never ran; check tune_every and epochs")
    head.load_state_dict(best_head)
    backbone.module.load_state_dict(best_model)
    scored = _score(backbone, head, test_items, device, best_threshold)
    result = {
        "model": model_name,
        "task": task_name,
        "family": family or "",
        "pipeline": "RiNALMo+gradual-unfreeze-FT",
        "metric": "flexible(+/-1)_F1",
        "num_train": len(train_items),
        "num_val": len(valid_items),
        "num_test": scored["num_test"],
        "tuned_threshold": best_threshold,
        "best_val_F1": best_f1,
        "best_epoch": best_epoch,
        **{key: value for key, value in scored.items() if key != "num_test"},
    }
    result_path = destination / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"TEST median_F1={result['median_F1']:.4f} mean_F1={result['mean_F1']:.4f} "
        f"P={result['mean_precision']:.4f} R={result['mean_recall']:.4f} "
        f"thr={best_threshold} epoch={best_epoch}",
        flush=True,
    )
    return result_path


def _logits_train(backbone, head, sequence: str, device: torch.device) -> torch.Tensor:
    states = backbone.hidden(sequence, device)
    return head(states)[0]
