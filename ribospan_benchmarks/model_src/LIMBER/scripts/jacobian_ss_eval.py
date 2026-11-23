#!/usr/bin/env python3
"""
Zero-shot secondary-structure evaluation via categorical Jacobian contact maps.
Adapts the collaborator's pipeline to our HybridMambaForMaskedLM.

Modes:
  baseline  – full model on all 7 datasets
  ablation  – skip one layer at a time (5 small datasets by default)

Results:
  <out>/baseline/<dataset>_results.csv  + baseline_summary.json
  <out>/ablation/layer_<k>_<T|M>/<dataset>_results.csv  + ablation_summary.json
"""

import sys, json, argparse, fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "new_inf"))

from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM
import categorical_jarcobian_evaluation as _cj_mod
# Silence the per-position trange that spams logs with thousands of progress bars
_cj_mod.trange = lambda *a, **kw: range(*a)
from categorical_jarcobian_evaluation import compute_cj_contact_matrix, compute_auprc

# Our vocabulary (single-nuc tokenizer)
# T treated as U; N mapped to its own token (id=9)
VOCAB = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "A": 5, "U": 6, "C": 7, "G": 8, "N": 9,
    "T": 6,   # RNA: T -> U
}

DATA_DIR  = REPO / "new_inf"
DATASETS  = ["CRW", "PDB", "RFAM", "RNP", "SRP", "tmRNA", "tRNAdb"]


# ── Model wrapper ────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    Wraps HybridMambaForMaskedLM so the Jacobian code can call model(x)['logits'].
    skip_layer=k  → ablation: run all layers except k
    skip_layer=None → full model
    """
    def __init__(self, model, skip_layer=None):
        self.model   = model
        self.skip_layer = skip_layer

    def __call__(self, input_ids, attention_mask=None):
        if self.skip_layer is None:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return {"logits": out.logits}

        # Manual forward with layer skip
        hidden   = self.model.embeddings(input_ids=input_ids)
        ext_mask = None
        if attention_mask is not None:
            ext_mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype))
            ext_mask = ext_mask * torch.finfo(hidden.dtype).min

        for i, (layer, is_m) in enumerate(zip(self.model.layers, self.model.is_mamba)):
            if i == self.skip_layer:
                continue
            hidden = layer(hidden) if is_m else layer(hidden, attention_mask=ext_mask)[0]

        return {"logits": self.model.cls(hidden)}

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self


# ── Utilities ────────────────────────────────────────────────────────────────

def load_model(ckpt_dir):
    cd  = json.load(open(Path(ckpt_dir) / "model_config.json"))
    cfg = HybridMambaConfig(
        layer_pattern                = cd["layer_pattern"],
        mamba_d_state                = cd["mamba_d_state"],
        mamba_d_conv                 = cd["mamba_d_conv"],
        mamba_expand                 = cd["mamba_expand"],
        vocab_size                   = cd["vocab_size"],
        hidden_size                  = cd["hidden_size"],
        num_attention_heads          = cd["num_attention_heads"],
        intermediate_size            = cd["intermediate_size"],
        max_position_embeddings      = cd["max_position_embeddings"],
        type_vocab_size              = cd["type_vocab_size"],
        hidden_act                   = cd["hidden_act"],
        hidden_dropout_prob          = cd["hidden_dropout_prob"],
        attention_probs_dropout_prob = cd["attention_probs_dropout_prob"],
        layer_norm_eps               = cd["layer_norm_eps"],
        position_embedding_type      = cd["position_embedding_type"],
    )
    model = HybridMambaForMaskedLM(cfg)
    state = torch.load(Path(ckpt_dir) / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    return model, cfg.layer_pattern


def load_dataset(name, n_max=None, seed=42):
    path = DATA_DIR / f"bpRNA_1m_90_{name}_less512.cvs"
    df   = pd.read_csv(path)
    if n_max and len(df) > n_max:
        df = df.sample(n=n_max, random_state=seed).reset_index(drop=True)
    return df


def eval_dataset(df, wrapper, device, tag):
    records = []
    for idx, row in df.iterrows():
        seq = str(row["seq"]).upper()
        dbn = str(row["dbn"])
        if len(seq) != len(dbn):
            continue
        try:
            cm     = compute_cj_contact_matrix(seq, wrapper, VOCAB, device)
            result = compute_auprc(cm, dbn, min_seq_sep=4)
            records.append({
                "name":     row.get("name", ""),
                "id":       row.get("id", idx),
                "length":   len(seq),
                "NormAUPR": result["NormAUPR"],
            })
        except Exception as e:
            print(f"  [WARN] {tag} idx={idx} failed: {e}", flush=True)

        if len(records) % 50 == 0 and len(records) > 0:
            valid = [r["NormAUPR"] for r in records if not np.isnan(r["NormAUPR"])]
            mean  = f"{np.mean(valid):.4f}" if valid else "N/A"
            print(f"  [{tag}] {len(records)}/{len(df)}  mean={mean}", flush=True)

    return pd.DataFrame(records)


def run_eval(wrapper, device, out_dir, dataset_names, n_max_map, reuse=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in dataset_names:
        csv_path = out_dir / f"{name}_results.csv"
        if reuse and csv_path.exists():
            valid = pd.read_csv(csv_path)["NormAUPR"].dropna()
            if len(valid):
                summary[name] = {"mean_NormAUPR": float(valid.mean()), "n": len(valid)}
                print(f"  [{name}] reused {csv_path.name}  mean_NormAUPR={valid.mean():.4f}  n={len(valid)}", flush=True)
                continue
        df = load_dataset(name, n_max=n_max_map.get(name))
        print(f"\n  [{name}] {len(df)} seqs", flush=True)
        res = eval_dataset(df, wrapper, device, name)
        res.to_csv(out_dir / f"{name}_results.csv", index=False)
        if len(res):
            valid = res["NormAUPR"].dropna()
            mean  = float(valid.mean())
            summary[name] = {"mean_NormAUPR": mean, "n": len(valid)}
            print(f"  [{name}] mean_NormAUPR={mean:.4f}  n={len(valid)}", flush=True)
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     required=True)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--mode",     choices=["baseline", "ablation"], default="baseline")
    ap.add_argument("--max_rfam", type=int, default=1000,
                    help="max RFAM sequences in baseline (default 1000)")
    ap.add_argument("--max_small", type=int, default=None,
                    help="cap non-RFAM datasets in baseline")
    ap.add_argument("--abl_datasets", nargs="+",
                    default=["PDB", "RNP", "SRP", "tmRNA", "tRNAdb"],
                    help="datasets for ablation (default: 5 small ones)")
    ap.add_argument("--abl_max",  type=int, default=None,
                    help="max seqs per dataset in ablation")
    ap.add_argument("--layer_start", type=int, default=0,
                    help="ablation: first layer index to run (inclusive)")
    ap.add_argument("--layer_end", type=int, default=None,
                    help="ablation: last layer index to run (inclusive; default = last layer)")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[jacobian_eval] ckpt  : {args.ckpt}", flush=True)
    print(f"[jacobian_eval] mode  : {args.mode}", flush=True)
    print(f"[jacobian_eval] device: {device}", flush=True)

    model, layer_pattern = load_model(args.ckpt)
    model = model.to(device).eval()
    n_layers = len(layer_pattern)
    print(f"[jacobian_eval] {n_layers} layers: {layer_pattern}", flush=True)

    out_root = Path(args.out)

    if args.mode == "baseline":
        n_max = {"RFAM": args.max_rfam}
        if args.max_small:
            for ds in DATASETS:
                if ds != "RFAM":
                    n_max[ds] = args.max_small
        print("\n=== Baseline (full model, all datasets) ===", flush=True)
        wrapper = ModelWrapper(model, skip_layer=None)
        summary = run_eval(wrapper, device, out_root / "baseline", DATASETS, n_max)
        json.dump(summary, open(out_root / "baseline_summary.json", "w"), indent=2)
        print(f"\n[baseline] {summary}", flush=True)

    elif args.mode == "ablation":
        n_max   = {ds: args.abl_max for ds in args.abl_datasets}
        # Resume: per-dataset CSVs already on disk are reused. ablation_summary.json is
        # updated under a file lock so several jobs can run disjoint layer ranges.
        summary_path = out_root / "ablation_summary.json"
        lock_path = out_root / ".ablation_summary.lock"
        last = n_layers - 1 if args.layer_end is None else min(args.layer_end, n_layers - 1)
        for k in range(args.layer_start, last + 1):
            lt  = layer_pattern[k]
            tag = f"layer_{k:02d}_{lt}"
            print(f"\n=== Ablation: skip layer {k} ({lt}) ===", flush=True)
            wrapper = ModelWrapper(model, skip_layer=k)
            new_results = run_eval(wrapper, device,
                                   out_root / "ablation" / tag,
                                   args.abl_datasets, n_max, reuse=True)
            with open(lock_path, "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                all_sum = json.load(open(summary_path)) if summary_path.exists() else {}
                all_sum.setdefault(tag, {}).update(new_results)
                json.dump(all_sum, open(summary_path, "w"), indent=2)
                fcntl.flock(lk, fcntl.LOCK_UN)
        print("\n[ablation] Done.", flush=True)


if __name__ == "__main__":
    main()
