#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Full-atlas t-SNE and 2x2 panel figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze import load_ids, plot_scatter  # noqa: E402
from scripts.config import display_model_name, load_experiment_config  # noqa: E402
from scripts.extract import embeddings_subdir_for_pooling  # noqa: E402
from scripts.plotting.export import EXPORT_FORMATS  # noqa: E402
from scripts.plotting.long import load_long_projection  # noqa: E402
from scripts.plotting.panels import plot_panels  # noqa: E402
from scripts.plotting.rfam import load_rfam_projection  # noqa: E402
import pandas as pd  # noqa: E402


def _short(name: str) -> str:
    return display_model_name(name)


def stratified_indices(labels: list[str], n_max: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_type.setdefault(lab, []).append(i)
    n = len(labels)
    if n_max >= n:
        return np.arange(n)
    quotas: dict[str, int] = {}
    remaining = n_max
    types = sorted(by_type, key=lambda t: len(by_type[t]), reverse=True)
    for i, lab in enumerate(types):
        left_types = len(types) - i
        share = max(1, remaining // left_types)
        take = min(len(by_type[lab]), share)
        quotas[lab] = take
        remaining -= take
    while remaining > 0:
        progressed = False
        for lab in types:
            if quotas[lab] < len(by_type[lab]) and remaining > 0:
                quotas[lab] += 1
                remaining -= 1
                progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    chosen: list[int] = []
    for lab, idxs in by_type.items():
        take = quotas.get(lab, 0)
        if take <= 0:
            continue
        pick = rng.choice(idxs, size=take, replace=False)
        chosen.extend(int(x) for x in pick)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def maybe_tsne(
    x: np.ndarray,
    *,
    perplexity: float,
    metric: str,
    seed: int,
    n_jobs: int = 1,
) -> np.ndarray:
    perp = min(float(perplexity), max(5.0, (len(x) - 1) / 3.0))
    x64 = np.asarray(x, dtype=np.float64)
    try:
        from openTSNE import TSNE as OpenTSNE

        print(
            f"[t-SNE] openTSNE on {x64.shape[0]} x {x64.shape[1]} "
            f"(perplexity={perp}, metric={metric})",
            flush=True,
        )
        reducer = OpenTSNE(
            n_components=2,
            perplexity=perp,
            metric=metric,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=True,
        )
        return np.asarray(reducer.fit(x64))
    except ImportError:
        from sklearn.manifold import TSNE as SklearnTSNE

        print(
            f"[t-SNE] sklearn on {x64.shape[0]} x {x64.shape[1]} "
            f"(perplexity={perp}, metric={metric})",
            flush=True,
        )
        reducer = SklearnTSNE(
            n_components=2,
            perplexity=perp,
            metric=metric,
            random_state=seed,
            init="random",
            learning_rate="auto",
        )
        return reducer.fit_transform(x64)


def run_one(
    *,
    emb_dir: Path,
    out_dir: Path,
    model_name: str,
    pooling: str,
    layer: int,
    analysis_cfg: dict[str, Any],
    formats: tuple[str, ...],
    panels: bool,
    rfam_dir: Path | None = None,
    map_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    ids = load_ids(emb_dir / meta["ids_file"])
    emb = np.load(emb_dir / f"embeddings_layer{layer}.npy")
    if len(ids) != emb.shape[0]:
        raise RuntimeError(f"id/embedding length mismatch in {emb_dir}")

    labels_all = ids["rna_type"].astype(str).tolist()
    seed = int(analysis_cfg["seed"])

    use_scaler = bool(analysis_cfg.get("use_scaler", True))
    use_pca = bool(analysis_cfg.get("use_pca", True))
    x = np.asarray(emb, dtype=np.float32)
    if use_scaler:
        x = StandardScaler().fit_transform(x)
    pca_dim = 0
    if use_pca:
        pca_dim = min(int(analysis_cfg.get("pca_dim", 50)), x.shape[0] - 1, x.shape[1])
        if pca_dim >= 2:
            x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
        else:
            pca_dim = 0

    max_points = analysis_cfg.get("tsne_max_points")
    if max_points is not None and int(max_points) > 0 and len(x) > int(max_points):
        idx = stratified_indices(labels_all, int(max_points), seed)
    else:
        idx = np.arange(len(x))

    print(
        f"[viz] {model_name} pooling={pooling} reducer=tsne "
        f"n={len(idx)} dim={x.shape[1]} scaler={use_scaler} pca={pca_dim}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tsne_{pooling}_layer{layer}"
    coords_path = out_dir / f"{stem}_coords.npy"
    index_path = out_dir / f"{stem}_index.npy"

    reuse = bool(analysis_cfg.get("reuse_coords", True))
    if (
        reuse
        and coords_path.exists()
        and index_path.exists()
    ):
        saved_idx = np.load(index_path)
        coords = np.load(coords_path)
        if len(coords) == len(saved_idx) and (
            len(saved_idx) == len(idx) and np.array_equal(saved_idx, idx)
        ):
            print(f"[viz] {model_name} reuse saved t-SNE coords n={len(idx)}", flush=True)
            idx = saved_idx
        else:
            reuse = False
    else:
        reuse = False

    if not reuse:
        coords = maybe_tsne(
            x[idx],
            perplexity=float(analysis_cfg.get("tsne_perplexity", 30)),
            metric=str(analysis_cfg.get("tsne_metric", "euclidean")),
            seed=seed,
            n_jobs=int(analysis_cfg.get("tsne_n_jobs", 1)),
        )
        np.save(coords_path, coords)
        np.save(index_path, idx)

    labels = [labels_all[int(i)] for i in idx]

    title = _short(model_name)
    written = plot_scatter(
        coords,
        labels,
        title=title,
        out_path=out_dir / f"{stem}.svg",
        formats=formats,
    )

    panel_written: list[Path] = []
    if panels:
        rfam_coords = rfam_labels = rfam_order = None
        long_coords = long_labels = long_order = None
        long_threshold = int(analysis_cfg.get("long_length_threshold", 1024))
        if rfam_dir is not None:
            loaded = load_rfam_projection(rfam_dir, layer, map_df=map_df)
            if loaded is not None:
                rfam_coords, rfam_labels, rfam_order = loaded
            long_loaded = load_long_projection(rfam_dir, layer)
            if long_loaded is not None:
                long_coords, long_labels, long_order, long_threshold = long_loaded
        panel_written = plot_panels(
            coords,
            labels,
            title=title,
            out_path=out_dir / f"{stem}_panels.svg",
            formats=formats,
            long_coords=long_coords,
            long_labels=long_labels,
            long_order=long_order,
            long_threshold=int(long_threshold),
            rfam_coords=rfam_coords,
            rfam_labels=rfam_labels,
            rfam_order=rfam_order,
        )

    info = {
        "model": model_name,
        "pooling": pooling,
        "reducer": "tsne",
        "layer": layer,
        "n_total": int(len(labels_all)),
        "n_projected": int(len(idx)),
        "use_scaler": use_scaler,
        "use_pca": use_pca,
        "pca_dim": int(pca_dim),
        "embedding_dim": int(emb.shape[1]),
        "plots": [str(p) for p in written],
        "panels": [str(p) for p in panel_written],
    }
    (out_dir / f"{stem}_meta.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--pooling", nargs="+", default=["mean"])
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Embedding layer index. Default: each model's meta.json layers[-1].",
    )
    parser.add_argument("--formats", nargs="+", default=list(EXPORT_FORMATS))
    parser.add_argument("--panels", action="store_true", default=True)
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Stratified subsample size. Use 0 for all points.",
    )
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA before projection.")
    parser.add_argument(
        "--no-scaler",
        action="store_true",
        help="Skip StandardScaler (raw embeddings -> t-SNE).",
    )
    parser.add_argument(
        "--recompute-tsne",
        action="store_true",
        help="Recompute t-SNE even if saved coords exist.",
    )
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    output_root = (project_root / cfg["output_root"]).resolve()
    model_names = args.models or list(cfg["models"]["names"])
    analysis_cfg = dict(cfg.get("analysis", {}))

    if args.max_points is not None:
        analysis_cfg["tsne_max_points"] = int(args.max_points)
    analysis_cfg["use_pca"] = not bool(args.no_pca)
    analysis_cfg["use_scaler"] = not bool(args.no_scaler)
    analysis_cfg["reuse_coords"] = not bool(args.recompute_tsne)
    analysis_cfg.setdefault("seed", 42)
    analysis_cfg.setdefault("pca_dim", 50)
    analysis_cfg.setdefault("tsne_perplexity", 30)
    analysis_cfg.setdefault("tsne_metric", "euclidean")
    analysis_cfg.setdefault("tsne_n_jobs", 1)

    panels = False if args.no_panels else bool(args.panels)
    map_path = output_root / "rfam_mapping" / "seq_to_rfam.csv"
    map_df = pd.read_csv(map_path) if map_path.exists() else None
    for name in model_names:
        for pooling in args.pooling:
            emb_dir = output_root / embeddings_subdir_for_pooling(pooling) / name
            if not (emb_dir / "meta.json").exists():
                print(f"skip missing embeddings: {emb_dir}", flush=True)
                continue
            meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
            layer = int(args.layer) if args.layer is not None else int(meta["layers"][-1])
            out_dir = output_root / "analysis" / name
            rfam_dir = out_dir
            info = run_one(
                emb_dir=emb_dir,
                out_dir=out_dir,
                model_name=name,
                pooling=pooling,
                layer=layer,
                analysis_cfg=analysis_cfg,
                formats=tuple(args.formats),
                panels=panels,
                rfam_dir=rfam_dir,
                map_df=map_df,
            )
            print(
                f"wrote {name} {pooling}/tsne: "
                f"n={info['n_projected']}/{info['n_total']} "
                f"scaler={info['use_scaler']} pca={info['pca_dim']}",
                flush=True,
            )
            for path in info["plots"] + info["panels"]:
                print(f"  {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
