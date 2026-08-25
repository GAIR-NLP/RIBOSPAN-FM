#!/usr/bin/env python3
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Map dataset URS IDs to RNAcentral v26 Rfam family annotations."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (  # noqa: E402
    load_experiment_config,
    load_sequence_table,
    public_path,
    resolve_dataset_path,
)


def load_dataset_rows(
    dataset_path: Path,
    *,
    id_column: str,
    type_column: str,
    sequence_column: str,
) -> pd.DataFrame:
    rows = load_sequence_table(
        dataset_path,
        id_column=id_column,
        type_column=type_column,
        sequence_column=sequence_column,
    )
    return pd.DataFrame(rows)


def parse_evalue(raw: str) -> float:
    try:
        return float(raw)
    except Exception:
        return float("inf")


def stream_best_hits(
    annotations_path: Path,
    keep_urs: set[str],
) -> dict[str, dict[str, Any]]:
    """Keep the best (lowest E-value, then highest score) Rfam hit per URS."""
    best: dict[str, dict[str, Any]] = {}
    n_hits = 0
    with gzip.open(annotations_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            urs = parts[0]
            if urs not in keep_urs:
                continue
            n_hits += 1
            hit = {
                "seq_id": urs,
                "rfam_id": parts[1],
                "score": float(parts[2]) if parts[2] else None,
                "evalue": parse_evalue(parts[3]),
                "seq_start": int(parts[4]),
                "seq_stop": int(parts[5]),
                "model_start": int(parts[6]),
                "model_stop": int(parts[7]),
                "rfam_description": parts[8],
            }
            prev = best.get(urs)
            if prev is None:
                best[urs] = hit
                continue
            prev_e = prev["evalue"]
            cur_e = hit["evalue"]
            prev_score = prev["score"] if prev["score"] is not None else float("-inf")
            cur_score = hit["score"] if hit["score"] is not None else float("-inf")
            if (cur_e, -cur_score) < (prev_e, -prev_score):
                best[urs] = hit
    return best, n_hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/rfam_annotations.tsv.gz"),
    )
    args = parser.parse_args(argv)

    cfg, project_root = load_experiment_config(args.config)
    output_root = (project_root / cfg["output_root"]).resolve()
    data_cfg = cfg["data"]
    dataset_path = resolve_dataset_path(cfg, project_root)
    out_dir = output_root / "rfam_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset_rows(
        dataset_path,
        id_column=str(data_cfg.get("id_column", "id")),
        type_column=str(data_cfg.get("type_column", "rna_type")),
        sequence_column=str(data_cfg.get("sequence_column", "sequence")),
    )
    df["seq_id"] = df["seq_id"].astype(str)
    df["rna_type"] = df["rna_type"].astype(str)
    urs_mask = df["seq_id"].str.startswith("URS")
    keep_urs = set(df.loc[urs_mask, "seq_id"].tolist())
    print(
        f"dataset n={len(df)} urs={urs_mask.sum()} non_urs={(~urs_mask).sum()}",
        flush=True,
    )

    annotations_path = (project_root / args.annotations).resolve()
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"missing {annotations_path}; download RNAcentral v26 "
            "rfam_annotations.tsv.gz first"
        )
    print(f"scanning {annotations_path}", flush=True)
    best, n_hits = stream_best_hits(annotations_path, keep_urs)
    print(f"matched hits={n_hits} unique_urs_with_rfam={len(best)}", flush=True)

    # One row per dataset sequence (preserve order / allow missing).
    records = []
    for row in df.itertuples(index=False):
        sid = str(row.seq_id)
        hit = best.get(sid)
        records.append(
            {
                "seq_id": sid,
                "rna_type": str(row.rna_type),
                "length": int(row.length),
                "has_urs": sid.startswith("URS"),
                "has_rfam": hit is not None,
                "rfam_id": None if hit is None else hit["rfam_id"],
                "rfam_description": None if hit is None else hit["rfam_description"],
                "score": None if hit is None else hit["score"],
                "evalue": None if hit is None else hit["evalue"],
                "seq_start": None if hit is None else hit["seq_start"],
                "seq_stop": None if hit is None else hit["seq_stop"],
                "model_start": None if hit is None else hit["model_start"],
                "model_stop": None if hit is None else hit["model_stop"],
            }
        )
    map_df = pd.DataFrame(records)
    map_path = out_dir / "seq_to_rfam.csv"
    map_df.to_csv(map_path, index=False)

    mapped = map_df[map_df["has_rfam"]].copy()
    family_counts = Counter(mapped["rfam_id"].tolist())
    type_cov = (
        map_df.groupby("rna_type")
        .agg(
            n=("seq_id", "size"),
            n_mapped=("has_rfam", "sum"),
        )
        .reset_index()
    )
    type_cov["frac_mapped"] = type_cov["n_mapped"] / type_cov["n"]
    type_cov = type_cov.sort_values("n", ascending=False)

    summary = {
        "source_annotations": public_path(annotations_path, project_root),
        "rnacentral_release": "26.0",
        "selection_rule": "lowest e-value; tie-break higher bit score",
        "n_sequences": int(len(map_df)),
        "n_urs": int(urs_mask.sum()),
        "n_non_urs": int((~urs_mask).sum()),
        "n_annotation_rows_for_urs": int(n_hits),
        "n_mapped": int(mapped.shape[0]),
        "frac_mapped_all": float(mapped.shape[0] / len(map_df)),
        "frac_mapped_urs": float(mapped.shape[0] / max(int(urs_mask.sum()), 1)),
        "n_rfam_families": int(len(family_counts)),
        "top_families": [
            {
                "rfam_id": fam,
                "n": int(cnt),
                "description": mapped.loc[mapped["rfam_id"] == fam, "rfam_description"].iloc[0],
            }
            for fam, cnt in family_counts.most_common(40)
        ],
        "coverage_by_rna_type": type_cov.to_dict(orient="records"),
        "files": {
            "seq_to_rfam_csv": public_path(map_path, project_root),
            "summary_json": public_path(out_dir / "summary.json", project_root),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    type_cov.to_csv(out_dir / "coverage_by_rna_type.csv", index=False)
    print(json.dumps({k: summary[k] for k in (
        "n_sequences", "n_mapped", "frac_mapped_all", "frac_mapped_urs", "n_rfam_families"
    )}, indent=2))
    print(f"wrote {map_path}")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
