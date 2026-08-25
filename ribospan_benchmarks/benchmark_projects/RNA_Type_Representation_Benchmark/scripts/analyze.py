# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Leave-one-out cosine-kNN (full biotype, then panel-aligned subsets)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plotting.export import EXPORT_FORMATS, save_figure  # noqa: E402
from scripts.config import load_experiment_config  # noqa: E402


RNA_TYPE_ORDER: list[str] = [
    "tRNA",
    "rRNA",
    "tmRNA",
    "mRNA",
    "lncRNA",
    "antisense_RNA",
    "antisense",
    "circRNA",
    "snRNA",
    "snoRNA",
    "scaRNA",
    "miRNA",
    "pre_miRNA",
    "piRNA",
    "siRNA",
    "sRNA",
    "ncRNA",
    "misc_RNA",
    "scRNA",
    "SRP_RNA",
    "RNase_P_RNA",
    "RNase_MRP_RNA",
    "telomerase_RNA",
    "hammerhead_ribozyme",
    "ribozyme",
    "Y_RNA",
    "vault_RNA",
    "guide_RNA",
    "precursor_RNA",
    "other",
]

RNA_TYPE_COLORS: dict[str, str] = {
    "tRNA": "#0050a0",
    "rRNA": "#1a7fd4",
    "tmRNA": "#4aa3e8",
    "mRNA": "#007a2f",
    "lncRNA": "#2db34a",
    "antisense_RNA": "#6fdc7a",
    "antisense": "#9be89f",
    "circRNA": "#c4f0c6",
    "snRNA": "#5b1fa6",
    "snoRNA": "#7b3fc9",
    "scaRNA": "#a56ae0",
    "miRNA": "#c41414",
    "pre_miRNA": "#e85d04",
    "piRNA": "#ff006e",
    "siRNA": "#ff6b35",
    "sRNA": "#6b3a2a",
    "ncRNA": "#8f5a3c",
    "misc_RNA": "#b07d57",
    "scRNA": "#d4a373",
    "SRP_RNA": "#c9a000",
    "RNase_P_RNA": "#e6c200",
    "RNase_MRP_RNA": "#f0d84a",
    "telomerase_RNA": "#00a8a8",
    "hammerhead_ribozyme": "#b0006d",
    "ribozyme": "#d90479",
    "Y_RNA": "#4a4a4a",
    "vault_RNA": "#6e6e6e",
    "guide_RNA": "#00838f",
    "precursor_RNA": "#ff8c42",
    "other": "#222222",
}

PLOT_EXCLUDE_TYPES: set[str] = {"NULL"}
FALLBACK_COLORS: list[str] = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
    "#2f9e44",
]


def ordered_rna_types(labels: list[str]) -> list[str]:
    present = {lab for lab in labels if lab not in PLOT_EXCLUDE_TYPES}
    ordered = [
        name for name in RNA_TYPE_ORDER if name in present and name != "other"
    ]
    leftovers = sorted(present - set(ordered) - {"other"})
    result = ordered + leftovers
    if "other" in present:
        result.append("other")
    return result


def color_for_rna_type(rna_type: str, fallback_index: int) -> str:
    if rna_type in RNA_TYPE_COLORS:
        return RNA_TYPE_COLORS[rna_type]
    return FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]


def load_ids(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_scatter(
    coords: np.ndarray,
    labels: list[str],
    *,
    title: str,
    out_path: Path,
    formats: tuple[str, ...] = EXPORT_FORMATS,
) -> list[Path]:
    """Atlas scatter in the same frame style as the composite figure."""
    # Lazy import avoids circular dependency (panels imports analyze).
    from scripts.plotting.panels import POINT_ALPHA, POINT_SIZE

    keep = np.array([lab not in PLOT_EXCLUDE_TYPES for lab in labels], dtype=bool)
    coords = np.asarray(coords)[keep]
    labels = [lab for lab, ok in zip(labels, keep) if ok]
    types = ordered_rna_types(labels)
    counts = {t: sum(1 for lab in labels if lab == t) for t in types}

    # Draw abundant / diffuse classes first; compact minority classes on top.
    draw_order = sorted(types, key=lambda t: counts[t], reverse=True)

    fig, ax = plt.subplots(figsize=(9.6, 8.4), layout="constrained")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    fallback_i = 0
    for rna_type in draw_order:
        mask = np.array([lab == rna_type for lab in labels])
        if not mask.any():
            continue
        color = color_for_rna_type(rna_type, fallback_i)
        if rna_type not in RNA_TYPE_COLORS:
            fallback_i += 1
        n = int(counts[rna_type])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=color,
            linewidths=0,
            rasterized=True,
            zorder=2 if n < 500 else 1,
        )

    handles = []
    handle_labels = []
    fallback_i = 0
    for rna_type in types:
        color = color_for_rna_type(rna_type, fallback_i)
        if rna_type not in RNA_TYPE_COLORS:
            fallback_i += 1
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=5.5,
                label=rna_type,
            )
        )
        handle_labels.append(rna_type)

    # Match composite / panel frame: square box, no ticks or axis labels.
    pad = 3.0
    xlim = (float(coords[:, 0].min()) - pad, float(coords[:, 0].max()) + pad)
    ylim = (float(coords[:, 1].min()) - pad, float(coords[:, 1].max()) + pad)
    x0, x1 = xlim
    y0, y1 = ylim
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0, 1e-6)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.9)

    ax.legend(
        handles=handles,
        labels=handle_labels,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.35,
        handletextpad=0.4,
        columnspacing=0.8,
        ncol=1,
        fontsize=8,
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


def _majority_label(row: np.ndarray) -> str:
    vals, counts = np.unique(row, return_counts=True)
    return str(vals[int(np.argmax(counts))])


def evaluate_supervised(
    x: np.ndarray,
    y: np.ndarray,
    *,
    knn_k: int,
    seed: int | None = None,
    test_size: float | None = None,
) -> dict[str, Any]:
    """Leave-one-out cosine-kNN accuracy and neighborhood purity."""
    del test_size, seed
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    n = int(len(y))
    if n < knn_k + 1:
        raise RuntimeError(f"need at least knn_k+1 samples for LOO kNN, got n={n}")

    print(
        f"  LOO kNN n={n} dim={x.shape[1]} k={knn_k} metric=cosine",
        flush=True,
    )
    nn = NearestNeighbors(n_neighbors=knn_k + 1, metric="cosine", n_jobs=-1)
    nn.fit(x)
    # Column 0 is self (distance 0); use the next k neighbors.
    neigh_idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    neigh_labels = y[neigh_idx]

    purity_point = (neigh_labels == y[:, None]).mean(axis=1).astype(np.float64)
    purity_mean = float(purity_point.mean())
    knn_pred = np.array([_majority_label(row) for row in neigh_labels], dtype=object)
    knn_acc = float((knn_pred == y).mean())

    per_class_purity: dict[str, float] = {}
    for lab in sorted(set(y.tolist())):
        mask = y == lab
        if not np.any(mask):
            continue
        per_class_purity[str(lab)] = float(purity_point[mask].mean())
    purity_macro = (
        float(np.mean(list(per_class_purity.values()))) if per_class_purity else float("nan")
    )
    print(
        f"  kNN LOO done acc={knn_acc:.4f} purity_mean={purity_mean:.4f} "
        f"purity_macro={purity_macro:.4f}",
        flush=True,
    )

    classes = sorted(set(y.tolist()))
    return {
        "knn": {
            "k": knn_k,
            "protocol": "leave_one_out",
            "metric": "cosine",
            "accuracy": knn_acc,
            "purity_mean": purity_mean,
            "purity_macro": purity_macro,
            "purity_per_class": per_class_purity,
            "report": classification_report(
                y, knn_pred, labels=classes, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y, knn_pred, labels=classes).tolist(),
            "labels": classes,
        },
        "n_eval": n,
    }


def analyze_one(
    *,
    emb_dir: Path,
    layer: int,
    analysis_cfg: dict[str, Any],
    out_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    ids = load_ids(emb_dir / meta["ids_file"])
    emb = np.load(emb_dir / f"embeddings_layer{layer}.npy")
    if len(ids) != emb.shape[0]:
        raise RuntimeError(f"id/embedding length mismatch in {emb_dir}")

    labels_all = ids["rna_type"].astype(str).tolist()
    counts = pd.Series(labels_all).value_counts()
    min_count = int(analysis_cfg.get("min_type_count", 1))
    keep_types = set(counts[counts >= min_count].index.tolist())
    keep_mask = np.array([lab in keep_types for lab in labels_all], dtype=bool)

    # Same space as t-SNE plots: raw mean-pool embeddings (no StandardScaler / PCA).
    x_all = np.asarray(emb, dtype=np.float32)
    print(
        f"[{model_name}] raw metrics on {int(keep_mask.sum())}/{len(labels_all)} "
        f"x {x_all.shape[1]} (min_count>={min_count})",
        flush=True,
    )

    x_sup = x_all[keep_mask]
    y_sup_labels = np.array(labels_all, dtype=object)[keep_mask]
    types = sorted(set(y_sup_labels.tolist()))

    supervised = evaluate_supervised(
        x_sup,
        y_sup_labels,
        knn_k=int(analysis_cfg["knn_k"]),
        seed=int(analysis_cfg["seed"]),
        test_size=float(analysis_cfg.get("test_size", 0.3)),
    )

    result = {
        "model": model_name,
        "layer": layer,
        "n_sequences": int(emb.shape[0]),
        "n_supervised": int(keep_mask.sum()),
        "min_type_count": min_count,
        "dropped_types": sorted(set(labels_all) - keep_types),
        "embedding_dim": int(emb.shape[1]),
        "metric_space": "raw",
        "use_scaler": False,
        "use_pca": False,
        "pca_dim": 0,
        "context_window": meta.get("context_window"),
        "types": types,
        "supervised": supervised,
    }
    with (out_dir / f"metrics_layer{layer}.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def upsert_summary_rows(
    path: Path, updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge ``updates`` by ``(model, layer)``, keeping extra subset fields."""

    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, int]] = []
    for row in existing:
        key = (str(row["model"]), int(row["layer"]))
        by_key[key] = dict(row)
        order.append(key)
    for row in updates:
        key = (str(row["model"]), int(row["layer"]))
        if key in by_key:
            by_key[key].update(row)
        else:
            by_key[key] = dict(row)
            order.append(key)
    merged = [by_key[key] for key in order]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(merged).to_csv(path.with_suffix(".csv"), index=False)
    return merged


def _functional_class_label(rna_type: str) -> str | None:
    from scripts.plotting.panels import CODING_TYPES, HOUSEKEEPING_TYPES, REGULATORY_TYPES

    if rna_type in HOUSEKEEPING_TYPES:
        return "housekeeping"
    if rna_type in REGULATORY_TYPES:
        return "regulatory"
    if rna_type in CODING_TYPES:
        return "coding"
    return None


def _eval_subset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    analysis_cfg: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    if len(y) < 50:
        print(f"  skip {name}: too few samples n={len(y)}", flush=True)
        return None
    counts = pd.Series(y).value_counts()
    min_count = int(analysis_cfg.get("min_type_count", 20))
    keep = set(counts[counts >= min_count].index.tolist())
    mask = np.array([lab in keep for lab in y], dtype=bool)
    if int(mask.sum()) < 50 or len(keep) < 2:
        print(
            f"  skip {name}: after min_count>={min_count} "
            f"n={int(mask.sum())} classes={len(keep)}",
            flush=True,
        )
        return None
    # LOO kNN needs at least 2 samples per class.
    y_k = y[mask]
    x_k = x[mask]
    vc = pd.Series(y_k).value_counts()
    ok = set(vc[vc >= 2].index.tolist())
    mask2 = np.array([lab in ok for lab in y_k], dtype=bool)
    x_k, y_k = x_k[mask2], y_k[mask2]
    if len(set(y_k.tolist())) < 2:
        print(f"  skip {name}: <2 classes after filters", flush=True)
        return None
    print(
        f"  [{name}] n={len(y_k)} classes={len(set(y_k.tolist()))}",
        flush=True,
    )
    supervised = evaluate_supervised(
        x_k,
        y_k,
        knn_k=int(analysis_cfg["knn_k"]),
        seed=int(analysis_cfg["seed"]),
        test_size=float(analysis_cfg.get("test_size", 0.3)),
    )
    return {
        "label_space": name,
        "n_sequences": int(len(y_k)),
        "n_classes": int(len(set(y_k.tolist()))),
        "classes": sorted(set(y_k.tolist())),
        "knn_accuracy": supervised["knn"]["accuracy"],
        "knn_purity_mean": supervised["knn"]["purity_mean"],
        "knn_purity_macro": supervised["knn"]["purity_macro"],
        "supervised": supervised,
    }


def analyze_subsets_one(
    *,
    emb_dir: Path,
    layer: int,
    analysis_cfg: dict[str, Any],
    out_dir: Path,
    model_name: str,
    map_df: pd.DataFrame,
    top_k_families: int,
) -> dict[str, Any]:
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    ids = load_ids(emb_dir / meta["ids_file"])
    emb = np.load(emb_dir / f"embeddings_layer{layer}.npy")
    if len(ids) != emb.shape[0] or len(ids) != len(map_df):
        raise RuntimeError(
            f"length mismatch model={model_name}: "
            f"ids={len(ids)} emb={emb.shape[0]} map={len(map_df)}"
        )

    x_all = np.asarray(emb, dtype=np.float32)
    rna_types = ids["rna_type"].astype(str).to_numpy()
    if "length" not in ids.columns:
        raise RuntimeError(f"ids.jsonl missing length for {model_name}")
    lengths = ids["length"].astype(int).to_numpy()
    long_threshold = int(analysis_cfg.get("long_length_threshold", 1024))

    functional_y = []
    functional_idx = []
    for i, t in enumerate(rna_types):
        g = _functional_class_label(t)
        if g is not None:
            functional_y.append(g)
            functional_idx.append(i)
    functional_classes = _eval_subset(
        x_all[np.asarray(functional_idx)],
        np.asarray(functional_y, dtype=object),
        analysis_cfg=analysis_cfg,
        name="functional_classes",
    )

    from scripts.plotting.panels import REGULATORY_TYPES

    reg_mask = np.array([t in REGULATORY_TYPES for t in rna_types], dtype=bool)
    regulatory = _eval_subset(
        x_all[reg_mask],
        rna_types[reg_mask],
        analysis_cfg=analysis_cfg,
        name="regulatory",
    )

    long_mask = lengths > long_threshold
    long = _eval_subset(
        x_all[long_mask],
        rna_types[long_mask],
        analysis_cfg=analysis_cfg,
        name=f"long_gt_{long_threshold}",
    )
    if long is not None:
        long["long_length_threshold"] = long_threshold
        long["n_raw_long"] = int(long_mask.sum())

    mapped = map_df["has_rfam"].to_numpy(dtype=bool)
    idx_all = np.flatnonzero(mapped)
    rfam_ids_all = map_df.loc[mapped, "rfam_id"].astype(str).tolist()
    from scripts.plotting.rfam import _display_labels

    labels, keep_mask, order = _display_labels(rfam_ids_all, top_k=top_k_families)
    idx = idx_all[keep_mask]
    rfam = _eval_subset(
        x_all[idx],
        np.asarray(labels, dtype=object),
        analysis_cfg={**analysis_cfg, "min_type_count": 1},
        name=f"rfam_top{top_k_families}",
    )
    if rfam is not None:
        rfam["display_labels"] = order
        rfam["top_k_families"] = int(top_k_families)

    result = {
        "model": model_name,
        "layer": layer,
        "metric_space": "raw",
        "panel_aligned": True,
        "long_length_threshold": long_threshold,
        "functional_classes": functional_classes,
        "long": long,
        "regulatory": regulatory,
        "rfam": rfam,
    }
    out_path = out_dir / f"metrics_layer{layer}_subsets.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _acc(block: dict[str, Any] | None, key: str = "knn_accuracy") -> float | None:
    if block is None:
        return None
    return float(block[key])


def _n(block: dict[str, Any] | None) -> int | None:
    if block is None:
        return None
    return int(block["n_sequences"])


def run_full(args: argparse.Namespace) -> None:
    cfg, project_root = load_experiment_config(args.config)
    output_root = (project_root / cfg["output_root"]).resolve()
    model_names = args.models or list(cfg["models"]["names"])
    analysis_cfg = cfg["analysis"]
    from scripts.extract import embeddings_subdir_for_pooling

    pooling = str(args.pooling)
    emb_root = embeddings_subdir_for_pooling(pooling)

    summaries = []
    for name in model_names:
        emb_dir = output_root / emb_root / name
        if not (emb_dir / "meta.json").exists():
            print(f"skip missing embeddings: {emb_dir}", flush=True)
            continue
        meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
        layers = [args.layer] if args.layer is not None else list(meta["layers"])
        out_dir = output_root / "analysis" / name
        if pooling != "mean":
            out_dir = output_root / "analysis" / f"{name}__{pooling}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            result = analyze_one(
                emb_dir=emb_dir,
                layer=int(layer),
                analysis_cfg=analysis_cfg,
                out_dir=out_dir,
                model_name=f"{name}[{pooling}]",
            )
            summaries.append(
                {
                    "model": name,
                    "pooling": pooling,
                    "layer": int(layer),
                    "knn_accuracy": result["supervised"]["knn"]["accuracy"],
                    "knn_purity_mean": result["supervised"]["knn"]["purity_mean"],
                    "knn_purity_macro": result["supervised"]["knn"]["purity_macro"],
                }
            )
            print(
                f"{name}[{pooling}] layer={layer}: "
                f"kNN={result['supervised']['knn']['accuracy']:.3f} "
                f"purity={result['supervised']['knn']['purity_mean']:.3f}"
            )

    summary_path = output_root / "analysis" / f"summary_{pooling}.json"
    upsert_summary_rows(summary_path, summaries)
    print(f"wrote {summary_path}")


def run_subsets(args: argparse.Namespace) -> None:
    from scripts.extract import embeddings_subdir_for_pooling

    cfg, project_root = load_experiment_config(args.config)
    output_root = (project_root / cfg["output_root"]).resolve()
    model_names = args.models or list(cfg["models"]["names"])
    analysis_cfg = dict(cfg["analysis"])
    pooling = str(args.pooling)
    emb_root = embeddings_subdir_for_pooling(pooling)
    long_threshold = int(analysis_cfg.get("long_length_threshold", 1024))

    map_path = output_root / "rfam_mapping" / "seq_to_rfam.csv"
    if not map_path.exists():
        raise FileNotFoundError(
            f"missing {map_path}; run python run_rna_type.py rfam first"
        )
    map_df = pd.read_csv(map_path)

    summary_path = output_root / "analysis" / f"summary_{pooling}.json"
    if summary_path.exists():
        base_rows = {
            (r["model"], int(r["layer"])): dict(r)
            for r in json.loads(summary_path.read_text(encoding="utf-8"))
        }
    else:
        base_rows = {}

    drop_prefixes = ("logreg_",)
    drop_exact = {
        "logreg_accuracy",
        "knn_housekeeping",
        "purity_housekeeping",
        "n_housekeeping",
        "knn_groups",
        "purity_groups",
        "n_groups",
    }
    updates: list[dict[str, Any]] = []
    for name in model_names:
        emb_dir = output_root / emb_root / name
        if not (emb_dir / "meta.json").exists():
            print(f"skip missing embeddings: {emb_dir}", flush=True)
            continue
        meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
        layers = [args.layer] if args.layer is not None else list(meta["layers"])
        out_dir = output_root / "analysis" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            print(f"[{name}] subset probes layer={layer}", flush=True)
            result = analyze_subsets_one(
                emb_dir=emb_dir,
                layer=int(layer),
                analysis_cfg=analysis_cfg,
                out_dir=out_dir,
                model_name=name,
                map_df=map_df,
                top_k_families=int(args.top_k_families),
            )
            key = (name, int(layer))
            row = base_rows.get(
                key,
                {"model": name, "pooling": pooling, "layer": int(layer)},
            )
            row = {
                k: v
                for k, v in row.items()
                if k not in drop_exact and not any(str(k).startswith(p) for p in drop_prefixes)
            }
            row.update(
                {
                    "model": name,
                    "pooling": pooling,
                    "layer": int(layer),
                    "long_length_threshold": long_threshold,
                    "knn_functional_classes": _acc(
                        result["functional_classes"], "knn_accuracy"
                    ),
                    "purity_functional_classes": _acc(
                        result["functional_classes"], "knn_purity_mean"
                    ),
                    "n_functional_classes": _n(result["functional_classes"]),
                    "knn_long": _acc(result["long"], "knn_accuracy"),
                    "purity_long": _acc(result["long"], "knn_purity_mean"),
                    "n_long": _n(result["long"]),
                    "knn_regulatory": _acc(result["regulatory"], "knn_accuracy"),
                    "purity_regulatory": _acc(result["regulatory"], "knn_purity_mean"),
                    "n_regulatory": _n(result["regulatory"]),
                    "knn_rfam_top20": _acc(result["rfam"], "knn_accuracy"),
                    "purity_rfam_top20": _acc(result["rfam"], "knn_purity_mean"),
                    "n_rfam_top20": _n(result["rfam"]),
                }
            )
            base_rows[key] = row
            updates.append(row)

            def _fmt(v: float | None) -> str:
                return f"{v:.3f}" if v is not None else "NA"

            print(
                f"{name} layer={layer}: "
                f"functional={_fmt(row.get('knn_functional_classes'))}/"
                f"{_fmt(row.get('purity_functional_classes'))} "
                f"long={_fmt(row.get('knn_long'))}/{_fmt(row.get('purity_long'))} "
                f"reg={_fmt(row.get('knn_regulatory'))}/{_fmt(row.get('purity_regulatory'))} "
                f"rfam={_fmt(row.get('knn_rfam_top20'))}/{_fmt(row.get('purity_rfam_top20'))}",
                flush=True,
            )

    upsert_summary_rows(summary_path, updates)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {summary_path.with_suffix('.csv')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--pooling",
        default="mean",
        help="Embedding dir: mean -> embeddings/, cls -> embeddings_cls/",
    )
    parser.add_argument("--top-k-families", type=int, default=20)
    args = parser.parse_args(argv)
    run_full(args)
    run_subsets(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
