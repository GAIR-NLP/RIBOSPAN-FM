# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot masked-marginal scoring of RNAGym DMS assays."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backends import backend_runtime_available, load_encoder  # noqa: E402
from scripts.config import (  # noqa: E402
    load_dataset_registry,
    load_experiment_config,
    load_tables,
    public_path,
    resolve_dataset_ids,
    results_dir,
    table_dir,
)
from scripts.logits import MLMScorer, build_scorer  # noqa: E402
from scripts.metrics import assay_metrics  # noqa: E402
from scripts.model_io import dtype_from_name, load_model_registry, resolve_device, weights_available  # noqa: E402
from scripts.mutations import PointMutation, mutation_window, parse_mutant  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _score_log_odds(
    log_probs: torch.Tensor,
    *,
    batch_index: int,
    token_index: int,
    wt_id: int,
    mt_id: int,
) -> float:
    return float(log_probs[batch_index, token_index, mt_id] - log_probs[batch_index, token_index, wt_id])


def _adaptive_batch_size(n_tokens: int, default: int) -> int:
    """Keep 10K-window forwards from stacking on a 24–48 GB card.

    Combinatorial ribozyme assays are short (≤100 nt) but have 1e5 unique
    mask patterns; a larger batch is the difference between minutes and hours.
    """

    if n_tokens > 4000:
        return 1
    if n_tokens > 2000:
        return min(2, default)
    if n_tokens > 1200:
        return min(4, default)
    if n_tokens <= 96:
        return max(default, 64)
    if n_tokens <= 256:
        return max(default, 32)
    return max(1, default)


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text


def _log_probs_with_oom_retry(scorer: MLMScorer, tokens: torch.Tensor) -> torch.Tensor:
    try:
        return scorer.log_probs(tokens)
    except Exception as exc:
        if not _is_oom(exc) or tokens.shape[0] == 1:
            raise
        torch.cuda.empty_cache()
        parts = []
        for row in tokens:
            parts.append(scorer.log_probs(row.unsqueeze(0)))
        return torch.cat(parts, dim=0)


def score_assay(
    *,
    scorer: MLMScorer,
    wt_sequence: str,
    mutants: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    scores = np.full(len(mutants), np.nan, dtype=np.float64)
    groups: dict[tuple[int, int, tuple[int, ...]], list[tuple[int, list[PointMutation]]]] = defaultdict(list)
    for row_index, raw in enumerate(mutants["mutant"].tolist()):
        parsed = parse_mutant(raw)
        if parsed is None:
            continue
        positions = [item.position for item in parsed]
        try:
            start, end = mutation_window(
                len(wt_sequence),
                positions,
                max_nucleotides=scorer.max_nucleotides,
            )
        except ValueError:
            continue
        key = (start, end, tuple(sorted(positions)))
        groups[key].append((row_index, parsed))

    items = list(groups.items())
    window_tokens = 0
    if items:
        start0, end0, _ = items[0][0]
        window_tokens = max(1, end0 - start0)
    step = _adaptive_batch_size(window_tokens, batch_size)
    for batch_start in range(0, len(items), step):
        chunk = items[batch_start : batch_start + step]
        token_list: list[torch.Tensor] = []
        meta: list[tuple[int, int, list[tuple[int, list[PointMutation]]]]] = []
        for (start, end, _), rows in chunk:
            window = wt_sequence[start:end]
            tokens = scorer.encode(window).clone()
            for pos in {item.position for _, parsed in rows for item in parsed}:
                tokens[0, scorer.offset + (pos - start)] = scorer.mask_id
            token_list.append(tokens)
            meta.append((start, end, rows))
        lengths = {int(t.shape[1]) for t in token_list}
        if len(lengths) != 1:
            # Mixed window lengths: fall back to per-row forwards.
            for tokens, (start, end, rows) in zip(token_list, meta):
                log_probs = _log_probs_with_oom_retry(scorer, tokens)
                _fill_group(scores, scorer, log_probs, 0, start, rows)
            continue
        batched = torch.cat(token_list, dim=0)
        log_probs = _log_probs_with_oom_retry(scorer, batched)
        for batch_index, (start, end, rows) in enumerate(meta):
            _fill_group(scores, scorer, log_probs, batch_index, start, rows)
    return scores


def _fill_group(
    scores: np.ndarray,
    scorer: MLMScorer,
    log_probs: torch.Tensor,
    batch_index: int,
    start: int,
    rows: list[tuple[int, list[PointMutation]]],
) -> None:
    for row_index, parsed in rows:
        total = 0.0
        ok = True
        for item in parsed:
            try:
                total += _score_log_odds(
                    log_probs,
                    batch_index=batch_index,
                    token_index=scorer.offset + (item.position - start),
                    wt_id=scorer.nt_id(item.wt),
                    mt_id=scorer.nt_id(item.mt),
                )
            except Exception:
                ok = False
                break
        if ok:
            scores[row_index] = total


def score_model_dataset(
    *,
    model_name: str,
    dataset_id: str,
    scorer: MLMScorer,
    mutants: pd.DataFrame,
    metadata: dict[str, Any],
    batch_size: int,
    out_dir: Path,
    skip_existing: bool,
) -> Path:
    dest = out_dir / dataset_id / model_name
    dest.mkdir(parents=True, exist_ok=True)
    metrics_path = dest / "metrics.json"
    if skip_existing and metrics_path.is_file():
        print(f"skip existing {dataset_id}/{model_name}", flush=True)
        return metrics_path
    wt = str(metadata["wt_sequence"])
    scores = score_assay(
        scorer=scorer,
        wt_sequence=wt,
        mutants=mutants,
        batch_size=batch_size,
    )
    scored = mutants.copy()
    scored["model_score"] = scores
    scored.to_csv(dest / "scores.csv", index=False)
    metrics = assay_metrics(scored["DMS_score"].to_numpy(), scores)
    for depth in ("single", "multiple"):
        mask = scored["depth"] == depth
        depth_metrics = assay_metrics(
            scored.loc[mask, "DMS_score"].to_numpy(),
            scored.loc[mask, "model_score"].to_numpy(),
        )
        for key, value in depth_metrics.items():
            metrics[f"{depth}_{key}"] = value
    payload = {
        "dataset_id": dataset_id,
        "model": model_name,
        "rna_type": metadata.get("rna_type"),
        "strategy": "masked-marginals",
        "wt_length": metadata.get("wt_length"),
        "metrics": metrics,
    }
    _write_json(metrics_path, payload)
    spearman = metrics.get("Spearman")
    spearman_s = (
        f"{spearman:.3f}"
        if isinstance(spearman, (int, float)) and spearman == spearman
        else "nan"
    )
    print(
        f"{model_name}/{dataset_id} Spearman={spearman_s} "
        f"n={metrics['n_scored']}/{metrics['n']}",
        flush=True,
    )
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--group", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    registry = load_dataset_registry(cfg, project_root)
    ids = resolve_dataset_ids(registry, group=args.group, datasets=args.datasets)
    specs = registry.get("datasets") or {}
    ids = sorted(
        ids,
        key=lambda name: (
            int((specs.get(name) or {}).get("wt_length") or 10**9),
            int((specs.get(name) or {}).get("n_mutants") or 0),
            name,
        ),
    )
    registry_path = (project_root / cfg["models_registry"]).resolve()
    model_registry = load_model_registry(
        registry_path,
        config_dir=registry_path.parent,
        project_root=project_root,
    )
    model_names = list(args.models or cfg.get("models", {}).get("names") or model_registry.keys())
    device = resolve_device(args.device or cfg.get("models", {}).get("device") or "cpu")
    dtype = dtype_from_name(str(cfg.get("models", {}).get("dtype") or "bfloat16"))
    skip_existing = not args.no_skip_existing
    batch_size = int(cfg.get("scoring", {}).get("batch_size") or 8)
    out_root = results_dir(cfg, project_root)

    for model_name in model_names:
        spec = model_registry[model_name]
        ok, reason = backend_runtime_available(spec.backend)
        if ok:
            ok, reason = weights_available(spec)
        if not ok:
            print(f"skip {model_name}: {reason}", flush=True)
            continue
        print(f"load {model_name} {public_path(spec.path, project_root)}", flush=True)
        loaded = load_encoder(spec, device=device, dtype=dtype)
        scorer = build_scorer(loaded)
        try:
            for dataset_id in tqdm(ids, desc=model_name):
                dest = table_dir(cfg, project_root, dataset_id)
                if not (dest / "metadata.json").is_file() or not (dest / "mutants.parquet").is_file():
                    print(f"skip {dataset_id}: missing frozen table", flush=True)
                    continue
                mutants, metadata = load_tables(dest)
                try:
                    score_model_dataset(
                        model_name=model_name,
                        dataset_id=dataset_id,
                        scorer=scorer,
                        mutants=mutants,
                        metadata=metadata,
                        batch_size=batch_size,
                        out_dir=out_root,
                        skip_existing=skip_existing,
                    )
                except Exception as exc:
                    err_dir = out_root / dataset_id / model_name
                    err_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(
                        err_dir / "error.json",
                        {
                            "dataset_id": dataset_id,
                            "model": model_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    print(f"FAIL {model_name}/{dataset_id}: {type(exc).__name__}: {exc}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        finally:
            loaded.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
