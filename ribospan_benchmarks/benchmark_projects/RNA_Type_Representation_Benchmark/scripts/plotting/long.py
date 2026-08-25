#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Long-sequence t-SNE (length > threshold)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze import PLOT_EXCLUDE_TYPES, load_ids, ordered_rna_types  # noqa: E402
from scripts.config import display_model_name, load_experiment_config  # noqa: E402
from scripts.extract import embeddings_subdir_for_pooling  # noqa: E402
from scripts.plotting.export import EXPORT_FORMATS, save_figure  # noqa: E402
from scripts.plotting.panels import draw_long_on_ax  # noqa: E402


def _short(name: str) -> str:
    return display_model_name(name)


def load_long_projection(
    out_dir: Path,
    layer: int,
) -> tuple[np.ndarray, list[str], list[str], int] | None:
    """Load saved long-seq t-SNE coords + labels + legend order + threshold."""
    stem = f"tsne_long_mean_layer{layer}"
    coords_path = out_dir / f"{stem}_coords.npy"
    index_path = out_dir / f"{stem}_index.npy"
    meta_path = out_dir / f"{stem}_meta.json"
    if not (coords_path.exists() and index_path.exists() and meta_path.exists()):
        return None
    coords = np.load(coords_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    labels = [str(x) for x in meta.get("point_labels", [])]
    order = [str(x) for x in meta.get("display_labels", [])]
    threshold = int(meta.get("long_length_threshold", 1024))
    if len(labels) != len(coords):
        return None
    if not order:
        order = ordered_rna_types(labels)
    return coords, labels, order, threshold


def plot_long_scatter(
    coords: np.ndarray,
    labels: list[str],
    label_order: list[str],
    *,
    title: str,
    threshold: int,
    out_path: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 8.4), layout="constrained")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    draw_long_on_ax(
        ax,
        coords,
        labels,
        label_order,
        title=title,
        threshold=threshold,
        size_scale=1.0,
        rasterized=True,
        show_legend=True,
        title_fontsize=13,
        legend_fontsize=7.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = save_figure(
        fig,
        out_path,
        formats,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return written


def run_one(
    *,
    emb_dir: Path,
    out_dir: Path,
    model_name: str,
    layer: int,
    analysis_cfg: dict[str, Any],
    formats: tuple[str, ...],
    reuse_coords: bool = True,
) -> dict[str, Any]:
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    ids = load_ids(emb_dir / meta["ids_file"])
    emb = np.load(emb_dir / f"embeddings_layer{layer}.npy")
    if len(ids) != emb.shape[0]:
        raise RuntimeError(
            f"length mismatch model={model_name}: ids={len(ids)} emb={emb.shape[0]}"
        )
    if "length" not in ids.columns:
        raise RuntimeError(f"ids.jsonl missing length column for {model_name}")

    threshold = int(analysis_cfg.get("long_length_threshold", 1024))
    min_count = int(analysis_cfg.get("min_type_count", 20))
    lengths = ids["length"].astype(int).to_numpy()
    rna_types = ids["rna_type"].astype(str).to_numpy()
    long_mask = np.array(
        [
            (L > threshold) and (lab not in PLOT_EXCLUDE_TYPES)
            for L, lab in zip(lengths, rna_types)
        ],
        dtype=bool,
    )
    # Match the long-seq kNN probe: drop rare biotypes (< min_type_count).
    from collections import Counter

    long_counts = Counter(rna_types[long_mask].tolist())
    keep_types = {lab for lab, n in long_counts.items() if n >= min_count}
    keep = np.array(
        [
            ok and (lab in keep_types)
            for ok, lab in zip(long_mask, rna_types)
        ],
        dtype=bool,
    )
    idx = np.flatnonzero(keep)
    if idx.size < 50:
        raise RuntimeError(
            f"too few long sequences (>{threshold}, min_count>={min_count}) "
            f"for {model_name}: {idx.size}"
        )
    labels = rna_types[idx].tolist()
    label_order = ordered_rna_types(labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tsne_long_mean_layer{layer}"
    coords_path = out_dir / f"{stem}_coords.npy"
    index_path = out_dir / f"{stem}_index.npy"

    need_recompute = (
        (not reuse_coords) or (not coords_path.exists()) or (not index_path.exists())
    )
    if not need_recompute:
        coords = np.load(coords_path)
        saved_idx = np.load(index_path)
        if not (len(coords) == len(idx) and np.array_equal(saved_idx, idx)):
            need_recompute = True

    if need_recompute:
        from scripts.plotting.atlas import maybe_tsne

        x = np.asarray(emb[idx], dtype=np.float32)
        seed = int(analysis_cfg.get("seed", 42))
        print(
            f"[long-viz] {model_name} compute t-SNE on length>{threshold} "
            f"n={len(idx)}/{len(ids)} types={len(label_order)} "
            f"(min_count>={min_count}; dropped={sorted(set(long_counts)-keep_types)})",
            flush=True,
        )
        coords = maybe_tsne(
            x,
            perplexity=float(analysis_cfg.get("tsne_perplexity", 30)),
            metric=str(analysis_cfg.get("tsne_metric", "euclidean")),
            seed=seed,
            n_jobs=int(analysis_cfg.get("tsne_n_jobs", 1)),
        )
        np.save(coords_path, coords)
        np.save(index_path, idx)
    else:
        coords = np.load(coords_path)
        print(
            f"[long-viz] {model_name} reuse long coords "
            f"n={len(idx)}/{len(ids)} length>{threshold}",
            flush=True,
        )

    written = plot_long_scatter(
        coords,
        labels,
        label_order,
        title=f"{_short(model_name)} (long RNAs >{threshold} nt)",
        threshold=threshold,
        out_path=out_dir / f"{stem}.svg",
        formats=formats,
    )
    info = {
        "model": model_name,
        "pooling": "mean",
        "label_space": "biotype_long",
        "layer": layer,
        "long_length_threshold": threshold,
        "min_type_count": min_count,
        "n_total": int(len(ids)),
        "n_long_raw": int(long_mask.sum()),
        "n_plotted": int(len(idx)),
        "dropped_rare_types": sorted(set(long_counts) - keep_types),
        "display_labels": label_order,
        "point_labels": labels,
        "plots": [str(p) for p in written],
    }
    (out_dir / f"{stem}_meta.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--formats", nargs="+", default=list(EXPORT_FORMATS))
    parser.add_argument(
        "--recompute-tsne",
        action="store_true",
        help="Recompute t-SNE instead of reusing saved coords.",
    )
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    output_root = (project_root / cfg["output_root"]).resolve()
    model_names = args.models or list(cfg["models"]["names"])
    analysis_cfg = dict(cfg.get("analysis", {}))

    for name in model_names:
        emb_dir = output_root / embeddings_subdir_for_pooling("mean") / name
        if not (emb_dir / "meta.json").exists():
            print(f"skip missing embeddings: {emb_dir}", flush=True)
            continue
        meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
        layer = int(meta["layers"][-1])
        out_dir = output_root / "analysis" / name
        info = run_one(
            emb_dir=emb_dir,
            out_dir=out_dir,
            model_name=name,
            layer=layer,
            analysis_cfg=analysis_cfg,
            formats=tuple(args.formats),
            reuse_coords=not bool(args.recompute_tsne),
        )
        print(
            f"[long-viz] wrote {name} n={info['n_plotted']} "
            f"thr={info['long_length_threshold']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
