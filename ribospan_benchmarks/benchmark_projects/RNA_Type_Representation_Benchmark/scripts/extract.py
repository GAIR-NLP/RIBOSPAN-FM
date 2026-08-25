# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Extract pooled sequence embeddings from registry models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backends import backend_runtime_available, load_encoder  # noqa: E402
from scripts.config import (  # noqa: E402
    load_experiment_config,
    load_sequence_table,
    public_path,
    resolve_dataset_path,
)
from scripts.model_io import (  # noqa: E402
    dtype_from_name,
    load_model_registry,
    resolve_device,
    weights_available,
)


def resolve_layers(requested: list[int], n_layers_including_embedding: int) -> list[int]:
    resolved = []
    for value in requested:
        layer = int(value)
        if layer < 0:
            layer = n_layers_including_embedding + layer
        if layer < 0 or layer >= n_layers_including_embedding:
            raise ValueError(
                f"layer {value} out of range for "
                f"{n_layers_including_embedding} states (0..{n_layers_including_embedding-1})"
            )
        resolved.append(layer)
    return resolved


def truncate_sequence(sequence: str, max_length: int, mode: str) -> tuple[str, bool]:
    if len(sequence) <= max_length:
        return sequence, False
    if mode == "prefix":
        return sequence[:max_length], True
    if mode == "center":
        start = max(0, (len(sequence) - max_length) // 2)
        return sequence[start : start + max_length], True
    raise ValueError(f"unsupported truncate mode: {mode}")


def normalize_poolings(pooling: str | list[str]) -> list[str]:
    if isinstance(pooling, str):
        values = [part.strip() for part in pooling.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in pooling]
    if not values:
        raise ValueError("pooling must be non-empty")
    allowed = {"mean", "cls", "mean_cls"}
    bad = [value for value in values if value not in allowed]
    if bad:
        raise ValueError(f"unsupported pooling(s): {bad}; allowed={sorted(allowed)}")
    return values


def embeddings_subdir_for_pooling(pooling: str) -> str:
    if pooling == "mean":
        return "embeddings"
    return f"embeddings_{pooling}"


def extract_for_model(
    *,
    model_name: str,
    registry,
    rows: list[dict[str, Any]],
    device,
    dtype,
    requested_layers: list[int],
    poolings: list[str],
    truncate_mode: str,
    output_root: Path,
    skip_existing: bool,
) -> list[Path]:
    out_dirs = {
        pooling: output_root / embeddings_subdir_for_pooling(pooling) / model_name
        for pooling in poolings
    }
    meta_paths = {pooling: out_dir / "meta.json" for pooling, out_dir in out_dirs.items()}

    pending = []
    for pooling in poolings:
        meta_path = meta_paths[pooling]
        if skip_existing and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta.get("n_sequences", -1)) == len(rows) and meta.get("pooling") == pooling:
                print(f"skip existing {model_name}/{pooling}: {meta_path}")
                continue
        pending.append(pooling)
    if not pending:
        return [meta_paths[pooling] for pooling in poolings]

    for pooling in pending:
        out_dirs[pooling].mkdir(parents=True, exist_ok=True)

    spec = registry[model_name]
    window = int(spec.trained_context_window)
    loaded = load_encoder(
        spec,
        device=device,
        dtype=dtype,
        requested_max_length=window,
    )
    # RNABert / RiboSpan / RNA-FM: embedding + N transformer layers.
    # RiNALMo: only final representation; index it as layer == num_hidden_layers.
    backend = (spec.backend or "rnabert").lower()
    n_states = loaded.num_hidden_layers + 1
    layers = resolve_layers(requested_layers, n_states)

    dim = int(loaded.hidden_size)
    arrays = {
        pooling: {
            layer: np.zeros((len(rows), dim), dtype=np.float32) for layer in layers
        }
        for pooling in pending
    }
    id_rows: list[str] = []
    n_truncated = 0
    effective_window = int(loaded.effective_max_length)
    for index, row in enumerate(
        tqdm(rows, desc=f"{model_name}[{backend}]@{effective_window}/{','.join(pending)}")
    ):
        seq, was_trunc = truncate_sequence(row["sequence"], effective_window, truncate_mode)
        n_truncated += int(was_trunc)
        vectors = loaded.embed_pooled(seq, layers=layers, poolings=pending)
        for pooling in pending:
            for layer, layer_vecs in vectors.items():
                arrays[pooling][layer][index] = layer_vecs[pooling]
        id_rows.append(
            json.dumps(
                {
                    "seq_id": row["seq_id"],
                    "rna_type": row["rna_type"],
                    "length": row["length"],
                    "used_length": len(seq),
                    "truncated": was_trunc or bool(row.get("store_truncated", False)),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    written: list[Path] = []
    for pooling in pending:
        out_dir = out_dirs[pooling]
        emb_paths = {layer: out_dir / f"embeddings_layer{layer}.npy" for layer in layers}
        id_path = out_dir / "ids.jsonl"
        for layer, arr in arrays[pooling].items():
            np.save(emb_paths[layer], arr)
        with id_path.open("w", encoding="utf-8") as id_handle:
            id_handle.writelines(id_rows)
        meta = {
            "model": model_name,
            "backend": backend,
            "model_path": public_path(spec.path, PROJECT_ROOT),
            "context_window": window,
            "effective_max_length": effective_window,
            "truncate": truncate_mode,
            "n_truncated_to_window": n_truncated,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "pooling": pooling,
            "layers": layers,
            "n_sequences": len(rows),
            "embedding_dim": dim,
            "embedding_files": {str(k): str(v.name) for k, v in emb_paths.items()},
            "ids_file": id_path.name,
        }
        meta_path = out_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        written.append(meta_path)
        print(f"wrote {meta_path}")

    loaded.close()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--pooling",
        default=None,
        help="Override config pooling. Comma-separated, e.g. mean,cls or cls",
    )
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    output_root = (project_root / cfg["output_root"]).resolve()
    data_cfg = cfg["data"]
    dataset_path = (
        Path(args.dataset).resolve()
        if args.dataset is not None
        else resolve_dataset_path(cfg, project_root)
    )
    rows = load_sequence_table(
        dataset_path,
        id_column=str(data_cfg.get("id_column", "id")),
        type_column=str(data_cfg.get("type_column", "rna_type")),
        sequence_column=str(data_cfg.get("sequence_column", "sequence")),
    )
    if not rows:
        raise SystemExit(f"empty dataset: {dataset_path}")

    model_cfg = cfg["models"]
    registry_path = (project_root / cfg["models_registry"]).resolve()
    registry = load_model_registry(
        registry_path,
        config_dir=registry_path.parent,
        project_root=project_root,
    )
    model_names = args.models or list(model_cfg["names"])
    device = resolve_device(args.device or model_cfg.get("device", "auto"))
    dtype = dtype_from_name(model_cfg.get("dtype", "bfloat16"))
    layers = list(model_cfg.get("layers", [-1]))
    poolings = normalize_poolings(args.pooling or model_cfg.get("pooling", "mean"))
    truncate_mode = str(model_cfg.get("truncate", "prefix"))
    skip_existing = bool(model_cfg.get("skip_existing", True))
    skip_log: list[dict[str, str]] = []

    for name in model_names:
        if name not in registry:
            raise KeyError(f"model {name} not in registry {registry_path}")
        spec = registry[name]
        available, reason = backend_runtime_available(spec.backend)
        if available:
            available, reason = weights_available(spec)
        if not available:
            record = {
                "model": name,
                "backend": spec.backend or "rnabert",
                "reason": reason,
            }
            skip_log.append(record)
            skip_dir = output_root / embeddings_subdir_for_pooling(poolings[0]) / name
            skip_dir.mkdir(parents=True, exist_ok=True)
            skip_path = skip_dir / "skipped.json"
            skip_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"skip {name} ({spec.backend}): {reason}", flush=True)
            continue
        extract_for_model(
            model_name=name,
            registry=registry,
            rows=rows,
            device=device,
            dtype=dtype,
            requested_layers=layers,
            poolings=poolings,
            truncate_mode=truncate_mode,
            output_root=output_root,
            skip_existing=skip_existing,
        )
    if skip_log:
        skip_summary = output_root / "extract_skips.json"
        skip_summary.write_text(json.dumps(skip_log, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {skip_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
