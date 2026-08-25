#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Rfam-family t-SNE (fit on top-k families only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze import load_ids  # noqa: E402
from scripts.config import display_model_name, load_experiment_config  # noqa: E402
from scripts.extract import embeddings_subdir_for_pooling  # noqa: E402
from scripts.plotting.export import EXPORT_FORMATS, save_figure  # noqa: E402
from scripts.plotting.panels import draw_rfam_on_ax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _short(name: str) -> str:
    return display_model_name(name)


def _rfam_sort_key(rfam_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in rfam_id if ch.isdigit())
    return (int(digits) if digits else 10**9, rfam_id)


def _display_labels(
    rfam_ids: list[str],
    *,
    top_k: int,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Keep only the top-k families; legend order is numeric RF id ascending."""
    counts: dict[str, int] = {}
    for fam in rfam_ids:
        counts[fam] = counts.get(fam, 0) + 1
    keep = {
        fam
        for fam, _ in sorted(counts.items(), key=lambda kv: (-kv[1], _rfam_sort_key(kv[0])))[
            :top_k
        ]
    }
    keep_mask = np.array([fam in keep for fam in rfam_ids], dtype=bool)
    labels = [fam for fam, ok in zip(rfam_ids, keep_mask) if ok]
    order = sorted(keep, key=_rfam_sort_key)
    return labels, keep_mask, order


def plot_rfam_scatter(
    coords: np.ndarray,
    labels: list[str],
    label_order: list[str],
    *,
    title: str,
    out_path: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 8.4), layout="constrained")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    draw_rfam_on_ax(
        ax,
        coords,
        labels,
        label_order,
        title=title,
        size_scale=1.0,
        rasterized=True,
        show_legend=True,
        title_fontsize=13,
        legend_fontsize=6.5,
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


def load_rfam_projection(
    rfam_dir: Path,
    layer: int,
    map_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str], list[str]] | None:
    """Load saved Rfam t-SNE coords + per-point labels + legend order."""
    stem = f"tsne_rfam_mean_layer{layer}"
    coords_path = rfam_dir / f"{stem}_coords.npy"
    index_path = rfam_dir / f"{stem}_index.npy"
    meta_path = rfam_dir / f"{stem}_meta.json"
    if not (coords_path.exists() and index_path.exists() and meta_path.exists()):
        return None
    coords = np.load(coords_path)
    idx = np.load(index_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    order = [str(x) for x in meta.get("display_labels", [])]
    if map_df is None:
        labels = [str(x) for x in meta.get("point_labels", [])]
        if len(labels) != len(coords):
            return None
    else:
        labels = map_df.loc[idx, "rfam_id"].astype(str).tolist()
    if not order:
        order = sorted(set(labels), key=_rfam_sort_key)
    return coords, labels, order


def run_one(
    *,
    emb_dir: Path,
    out_dir: Path,
    model_name: str,
    layer: int,
    map_df: pd.DataFrame,
    analysis_cfg: dict[str, Any],
    top_k_families: int,
    formats: tuple[str, ...],
    reuse_coords: bool = True,
) -> dict[str, Any]:
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    ids = load_ids(emb_dir / meta["ids_file"])
    emb = np.load(emb_dir / f"embeddings_layer{layer}.npy")
    if len(ids) != emb.shape[0] or len(ids) != len(map_df):
        raise RuntimeError(
            f"length mismatch model={model_name}: "
            f"ids={len(ids)} emb={emb.shape[0]} map={len(map_df)}"
        )

    mapped_mask = map_df["has_rfam"].to_numpy(dtype=bool)
    idx_all = np.flatnonzero(mapped_mask)
    if idx_all.size < 50:
        raise RuntimeError(f"too few Rfam-mapped points for {model_name}: {idx_all.size}")

    rfam_ids_all = map_df.loc[mapped_mask, "rfam_id"].astype(str).tolist()
    labels, keep_mask, label_order = _display_labels(
        rfam_ids_all, top_k=top_k_families
    )
    idx = idx_all[keep_mask]

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tsne_rfam_mean_layer{layer}"
    coords_path = out_dir / f"{stem}_coords.npy"
    index_path = out_dir / f"{stem}_index.npy"

    # t-SNE only on the plotted top-k families (dropped families must not
    # influence the layout).
    need_recompute = (
        (not reuse_coords)
        or (not coords_path.exists())
        or (not index_path.exists())
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
            f"[rfam-viz] {model_name} compute t-SNE on top-{top_k_families} "
            f"n={len(idx)}/{len(ids)} families={len(label_order)}",
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
        print(
            f"[rfam-viz] {model_name} reuse top-{top_k_families} coords "
            f"n={len(idx)}/{len(ids)}",
            flush=True,
        )

    written = plot_rfam_scatter(
        coords,
        labels,
        label_order,
        title=f"{_short(model_name)} (Rfam)",
        out_path=out_dir / f"{stem}.svg",
        formats=formats,
    )
    info = {
        "model": model_name,
        "pooling": "mean",
        "label_space": "rfam_family",
        "layer": layer,
        "n_total": int(len(ids)),
        "n_mapped_total": int(len(idx_all)),
        "n_plotted": int(len(idx)),
        "n_rfam_families_raw": int(len(set(rfam_ids_all))),
        "top_k_families_displayed": int(top_k_families),
        "exclude_other_rfam": True,
        "legend": "rfam_id_only",
        "display_labels": label_order,
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
    parser.add_argument("--top-k-families", type=int, default=20)
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

    map_path = output_root / "rfam_mapping" / "seq_to_rfam.csv"
    if not map_path.exists():
        raise FileNotFoundError(
            f"missing {map_path}; run python run_rna_type.py rfam first"
        )
    map_df = pd.read_csv(map_path)
    if "has_rfam" not in map_df.columns:
        raise RuntimeError("seq_to_rfam.csv missing has_rfam column")

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
            map_df=map_df,
            analysis_cfg=analysis_cfg,
            top_k_families=int(args.top_k_families),
            formats=tuple(args.formats),
            reuse_coords=not bool(args.recompute_tsne),
        )
        print(
            f"wrote {name}: plotted={info['n_plotted']}/{info['n_mapped_total']} "
            f"families_shown={len(info['display_labels'])}",
            flush=True,
        )
        for path in info["plots"]:
            print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
