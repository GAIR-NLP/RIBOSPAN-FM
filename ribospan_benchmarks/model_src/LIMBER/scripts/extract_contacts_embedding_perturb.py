#!/usr/bin/env python3
"""Embedding-perturbation contact extraction from a (frozen) RNA FM — kmer1.

For each sequence: capture the final-layer hidden states H_wt[L,d]. Then for
each position i, inject a perturbation at the chosen layer (default: embedding
output) ONLY at position i (via a forward hook, weights untouched), forward,
and capture the perturbed final hidden states. The interaction score is

    S[i,j] = || H_perturbed[j] - H_wt[j] ||_2

then symmetrize, zero diagonal, APC. Perturbation modes: gaussian (default,
epsilon-scaled fixed noise), zero (ablate position), learned_mask (replace the
input token with [MASK]). perturb_layer in {embedding, layer_3, layer_6, final}.

Resumable; first version is kmer1 only. Same output layout as the Jacobian
extractor.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc          # noqa: E402
from contact_metrics import evaluate  # noqa: E402


def _as_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


class Perturber:
    """Holds hooks: capture final hidden states; inject per-row perturbation."""

    def __init__(self, model, model_type, perturb_layer, mode, epsilon, hidden, device, seed=0):
        mm = cc.module_map(model, model_type)
        layers = mm["layers"]
        idx = {"embedding": None, "layer_3": 2, "layer_6": 5, "final": len(layers) - 1}
        self.perturb_module = mm["embedding"] if perturb_layer == "embedding" \
            else layers[min(idx[perturb_layer], len(layers) - 1)]
        self.response_module = layers[-1]
        self.mode, self.epsilon = mode, epsilon
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.pert_vec = (torch.randn(hidden, generator=g) * epsilon).to(device)
        self.captured = {}
        self.positions = None          # per-row token positions to perturb (or None)
        self._h1 = self.response_module.register_forward_hook(self._capture)
        self._h2 = self.perturb_module.register_forward_hook(self._inject)

    def _capture(self, module, inp, out):
        self.captured["h"] = _as_tensor(out).detach()

    def _inject(self, module, inp, out):
        if self.positions is None:
            return out
        h = _as_tensor(out)
        h = h.clone()
        for r, pos in enumerate(self.positions):
            if self.mode == "zero":
                h[r, pos, :] = 0.0
            else:                      # gaussian (learned_mask handled via input ids)
                h[r, pos, :] = h[r, pos, :] + self.pert_vec
        return _rewrap(out, h)

    def remove(self):
        self._h1.remove(); self._h2.remove()


@torch.no_grad()
def perturb_scores(model, tokenizer, seq, device, batch_size, perturb_layer,
                   mode, epsilon, model_type):
    L = len(seq)
    hidden = model.config.hidden_size
    wt_ids, am = cc.encode_kmer1(tokenizer, seq, device)     # [1, L+2]
    pert = Perturber(model, model_type, perturb_layer, mode, epsilon, hidden, device)

    # wild-type final hidden (no perturbation)
    pert.positions = None
    model(input_ids=wt_ids, attention_mask=am)
    H_wt = pert.captured["h"][0, 1:1 + L].float()            # [L, d]

    S = np.zeros((L, L), dtype=np.float64)
    mask_id = tokenizer.mask_token_id
    for s in range(0, L, batch_size):
        idxs = list(range(s, min(s + batch_size, L)))
        b = len(idxs)
        rows = wt_ids.repeat(b, 1).clone()                    # [b, L+2]
        if mode == "learned_mask":
            for r, i in enumerate(idxs):
                rows[r, i + 1] = mask_id
            pert.positions = None                             # input-level mask, no hook
        else:
            pert.positions = [i + 1 for i in idxs]            # token pos = base i + 1 (CLS)
        model(input_ids=rows, attention_mask=torch.ones_like(rows))
        H_p = pert.captured["h"][:, 1:1 + L].float()          # [b, L, d]
        delta = torch.linalg.norm(H_p - H_wt.unsqueeze(0), dim=-1).cpu().numpy()  # [b, L]
        for r, i in enumerate(idxs):
            S[i] = delta[r]
    pert.remove()
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_type", choices=["bert", "hybrid"], required=True)
    p.add_argument("--tokenizer_type", default="kmer1")
    p.add_argument("--vocab_dir", default="tokenizers/single")
    p.add_argument("--dataset_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max-seqs", type=int, default=100, dest="max_seqs")
    p.add_argument("--max-len", type=int, default=256, dest="max_len")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--num-plots", type=int, default=10, dest="num_plots")
    p.add_argument("--perturb-layer", default="embedding",
                   choices=["embedding", "layer_3", "layer_6", "final"], dest="perturb_layer")
    p.add_argument("--mode", default="gaussian", choices=["gaussian", "zero", "learned_mask"])
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.tokenizer_type != "kmer1":
        raise SystemExit("first version supports --tokenizer_type kmer1 only")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = cc.load_model(args.model_dir, args.model_type, device)
    tokenizer = cc.load_tokenizer(args.vocab_dir)

    out = Path(args.output_dir)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "example_plots").mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(args.dataset_jsonl)]
    recs = [r for r in recs if r["length"] <= args.max_len][:args.max_seqs]

    rows_csv, n_plot, skipped = [], 0, 0
    for k, rec in enumerate(recs):
        npy = out / "scores" / f"{rec['id']}.npy"
        if npy.exists():
            S_apc, S_sym = np.load(npy), None
        else:
            S_raw = perturb_scores(model, tokenizer, rec["sequence"], device,
                                   args.batch_size, args.perturb_layer, args.mode,
                                   args.epsilon, args.model_type)
            S_sym, S_apc = cc.finalize(S_raw)
            np.save(npy, S_apc)
        m = evaluate(S_apc, rec["pairs"], rec["sequence"])
        if m.get("skipped"):
            skipped += 1
            continue
        rows_csv.append({"id": rec["id"], "family": rec["family"],
                         "length": rec["length"], **m})
        if n_plot < args.num_plots and S_sym is not None:
            cc.plot_example(out / "example_plots" / f"{rec['id']}.png", rec,
                            S_sym, S_apc, auprc=m["AUPRC"], f1=m["best_F1"])
            n_plot += 1
        print(f"[{k+1}/{len(recs)}] {rec['id']:42} AUPRC={m['AUPRC']:.3f} "
              f"P@L={m['P_at_L']:.3f} F1={m['best_F1']:.3f}")

    if rows_csv:
        with open(out / "metrics_per_sequence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader(); w.writerows(rows_csv)

    def agg(key, fn):
        vals = [r[key] for r in rows_csv if r.get(key) == r.get(key)]
        return fn(vals) if vals else float("nan")

    summary = {"method": "embedding_perturb", "perturb_layer": args.perturb_layer,
               "mode": args.mode, "epsilon": args.epsilon,
               "model_dir": args.model_dir, "model_type": args.model_type,
               "dataset": args.dataset_jsonl, "num_sequences": len(rows_csv),
               "num_skipped": skipped}
    for key in ("AUPRC", "AUROC", "P_at_L", "P_at_L2", "P_at_num_gold",
                "best_F1", "best_precision", "best_recall",
                "canonical_pair_AUPRC", "canonical_pair_F1"):
        summary[f"mean_{key}"] = agg(key, statistics.mean)
        summary[f"median_{key}"] = agg(key, statistics.median)
    json.dump(summary, open(out / "metrics_summary.json", "w"), indent=2)

    print(f"\n[done] {len(rows_csv)} seqs ({skipped} skipped) -> {out}")
    print(f"  mean AUPRC={summary['mean_AUPRC']:.4f}  P@L={summary['mean_P_at_L']:.4f}  "
          f"best_F1={summary['mean_best_F1']:.4f}")


if __name__ == "__main__":
    main()
