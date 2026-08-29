# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Embed frozen mRNABench sequences with the RIBOSPAN benchmark encoder stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backends import backend_runtime_available, load_encoder  # noqa: E402
from scripts.chunked import embed_sequence_mean  # noqa: E402
from scripts.config import (  # noqa: E402
    embeddings_dir,
    load_dataset_registry,
    load_experiment_config,
    load_tables,
    public_path,
    resolve_dataset_ids,
    table_dir,
)
from scripts.model_io import (  # noqa: E402
    dtype_from_name,
    load_model_registry,
    resolve_device,
    weights_available,
)


def _existing_embeddings(out_dir: Path, model_name: str, n_rows: int) -> Path | None:
    dest = out_dir / model_name
    meta_path = dest / "meta.json"
    npy_path = dest / "embeddings.npy"
    if not (meta_path.is_file() and npy_path.is_file()):
        return None
    existing = np.load(npy_path, mmap_mode="r")
    if int(existing.shape[0]) != n_rows:
        return None
    return npy_path


def _write_embeddings(
    store: Path, model_name: str, embeddings: np.ndarray, meta: dict
) -> Path:
    dest = store / model_name
    dest.mkdir(parents=True, exist_ok=True)
    npy_path = dest / "embeddings.npy"
    np.save(npy_path, np.asarray(embeddings, dtype=np.float32))
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return npy_path


def extract_model_dataset(
    *,
    model_name: str,
    dataset_id: str,
    spec,
    loaded,
    sequences,
    metadata: dict,
    device,
    dtype,
    limit: int | None,
    out_dir: Path,
) -> Path:
    n_rows = int(len(sequences) if limit is None else min(limit, len(sequences)))
    dim = int(loaded.hidden_size)
    effective_max_length = int(loaded.effective_max_length)
    embeddings = np.zeros((n_rows, dim), dtype=np.float32)
    n_chunked = 0
    n_chunks_total = 0
    iterator = sequences["sequence"].tolist()[:n_rows]
    for index, sequence in enumerate(
        tqdm(iterator, desc=f"{model_name}/{dataset_id}@{effective_max_length}")
    ):
        vector, n_chunks = embed_sequence_mean(loaded, str(sequence))
        if vector.shape[0] != dim:
            raise RuntimeError(
                f"{model_name}: embedding dim {vector.shape} != hidden_size {dim}"
            )
        embeddings[index] = vector
        n_chunks_total += n_chunks
        n_chunked += int(n_chunks > 1)

    meta = {
        "model": model_name,
        "dataset_id": dataset_id,
        "backend": spec.backend,
        "model_path": public_path(spec.path, PROJECT_ROOT),
        "context_window": int(spec.trained_context_window),
        "effective_max_length": effective_max_length,
        "chunking": "concat_then_mean",
        "n_rows": n_rows,
        "n_chunked_sequences": n_chunked,
        "n_chunks_total": n_chunks_total,
        "embedding_dim": dim,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "pooling": "mean",
        "layer": "last",
        "n_table_rows": int(metadata.get("n_rows", n_rows)),
    }
    return _write_embeddings(out_dir, model_name, embeddings, meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--group", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N rows")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    ds_registry = load_dataset_registry(cfg, project_root)
    dataset_ids = resolve_dataset_ids(ds_registry, group=args.group, datasets=args.datasets)
    registry_path = (project_root / cfg["models_registry"]).resolve()
    model_registry = load_model_registry(
        registry_path,
        config_dir=registry_path.parent,
        project_root=project_root,
    )
    model_names = args.models or list(cfg["models"]["names"])
    device = resolve_device(args.device or cfg["models"].get("device", "auto"))
    dtype = dtype_from_name(cfg["models"].get("dtype", "bfloat16"))
    skip_existing = (not args.no_skip_existing) and bool(cfg["models"].get("skip_existing", True))

    jobs = []
    for dataset_id in dataset_ids:
        tables = table_dir(cfg, project_root, dataset_id)
        if not (tables / "metadata.json").is_file():
            print(f"missing tables for {dataset_id} under {tables}", flush=True)
            continue
        sequences, _, metadata = load_tables(tables)
        out_dir = embeddings_dir(cfg, project_root, dataset_id)
        jobs.append((dataset_id, out_dir, sequences, metadata))
    if not jobs:
        raise SystemExit("no frozen tables under data/; see README")

    skip_log: list[dict[str, str]] = []
    for model_name in model_names:
        if model_name not in model_registry:
            raise KeyError(f"model {model_name} not in {registry_path}")
        spec = model_registry[model_name]
        available, reason = backend_runtime_available(spec.backend)
        if available:
            available, reason = weights_available(spec)
        if not available:
            record = {"model": model_name, "backend": spec.backend or "", "reason": reason}
            skip_log.append(record)
            print(f"skip {model_name}: {reason}", flush=True)
            continue
        pending = []
        for dataset_id, out_dir, sequences, metadata in jobs:
            n_rows = int(
                len(sequences) if args.limit is None else min(args.limit, len(sequences))
            )
            existing = (
                _existing_embeddings(out_dir, model_name, n_rows) if skip_existing else None
            )
            if existing is not None:
                print(f"skip existing {model_name}/{dataset_id}: {existing}", flush=True)
                continue
            pending.append((dataset_id, out_dir, sequences, metadata))
        if not pending:
            continue
        loaded = load_encoder(
            spec,
            device=device,
            dtype=dtype,
            requested_max_length=int(spec.trained_context_window),
        )
        try:
            for dataset_id, out_dir, sequences, metadata in pending:
                extract_model_dataset(
                    model_name=model_name,
                    dataset_id=dataset_id,
                    spec=spec,
                    loaded=loaded,
                    sequences=sequences,
                    metadata=metadata,
                    device=device,
                    dtype=dtype,
                    limit=args.limit,
                    out_dir=out_dir,
                )
        finally:
            loaded.close()

    if skip_log:
        skip_path = (
            project_root / cfg.get("output_root", "outputs/mrnabench") / "extract_skips.json"
        )
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        skip_path.write_text(
            json.dumps(skip_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
