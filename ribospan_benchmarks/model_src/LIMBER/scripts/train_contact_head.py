#!/usr/bin/env python3
"""Train a supervised contact-prediction head on FROZEN LM embeddings (Part 6).

Reads cached base-level embeddings ({id}.pt) + train/val/test jsonl splits,
trains ContactPredictor with BCEWithLogitsLoss (pos_weight=auto for the heavy
class imbalance), selects the epoch with best val F1, evaluates on test, and
saves model / logs / per-seq test metrics / predictions / plots.
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
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from models.contact_head import ContactPredictor, valid_pair_mask  # noqa: E402
from contact_metrics import supervised_metrics  # noqa: E402


def load_split(jsonl, emb_dir, max_len):
    recs = [json.loads(l) for l in open(jsonl)]
    out = []
    for r in recs:
        if r["length"] > max_len:
            continue
        ep = Path(emb_dir) / f"{r['id']}.pt"
        if ep.exists():
            out.append(r)
    return out


def gold_target(L, pairs, device):
    G = torch.zeros((L, L), device=device)
    for i, j in pairs:
        G[i, j] = G[j, i] = 1.0
    return G


def emb(emb_dir, rid, device):
    return torch.load(Path(emb_dir) / f"{rid}.pt", weights_only=True).float().to(device)


@torch.no_grad()
def eval_split(model, recs, emb_dir, device, min_sep, threshold, want_preds=False):
    model.eval()
    rows, preds = [], {}
    for r in recs:
        H = emb(emb_dir, r["id"], device)
        prob = torch.sigmoid(model(H)).cpu().numpy()
        m = supervised_metrics(prob, r["pairs"], r["sequence"], min_sep, threshold)
        if m.get("skipped"):
            continue
        rows.append({"id": r["id"], "family": r["family"], "length": r["length"], **m})
        if want_preds:
            preds[r["id"]] = prob
    return rows, preds


def agg(rows, key, fn=statistics.mean):
    vals = [r[key] for r in rows if r.get(key) == r.get(key)]
    return fn(vals) if vals else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--pos-weight", default="auto", dest="pos_weight")
    p.add_argument("--max-len", type=int, default=512, dest="max_len")
    p.add_argument("--min-sep", type=int, default=4, dest="min_sep")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pair-dim", type=int, default=128, dest="pair_dim")
    p.add_argument("--hidden2d", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=8, dest="num_blocks")
    p.add_argument("--num-plots", type=int, default=10, dest="num_plots")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "example_plots").mkdir(parents=True, exist_ok=True)

    meta = json.load(open(Path(args.embedding_dir).parent / "metadata.json"))
    hidden_dim = meta["hidden_dim"]
    train = load_split(args.train_jsonl, args.embedding_dir, args.max_len)
    val = load_split(args.val_jsonl, args.embedding_dir, args.max_len)
    test = load_split(args.test_jsonl, args.embedding_dir, args.max_len)
    print(f"[train] split sizes train/val/test = {len(train)}/{len(val)}/{len(test)}  hidden_dim={hidden_dim}")

    # auto pos_weight = total_negatives / total_positives over train candidates
    if args.pos_weight == "auto":
        pos = neg = 0
        for r in train:
            L = r["length"]
            mask = valid_pair_mask(L, args.min_sep)
            n_cand = int(mask.sum())
            g = len({(min(i, j), max(i, j)) for i, j in r["pairs"] if abs(i - j) >= args.min_sep})
            pos += g; neg += n_cand - g
        pw = max(1.0, neg / max(1, pos))
    else:
        pw = float(args.pos_weight)
    print(f"[train] pos_weight={pw:.2f}")

    model = ContactPredictor(hidden_dim, args.pair_dim, args.hidden2d, args.num_blocks).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[train] contact head params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    log_rows, val_rows_log, best_f1, best_state = [], [], -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train))
        tot, nb = 0.0, 0
        for k in order:
            r = train[k]
            H = emb(args.embedding_dir, r["id"], device)
            L = r["length"]
            logits = model(H)
            mask = valid_pair_mask(L, args.min_sep, device)
            target = gold_target(L, r["pairs"], device)
            loss = lossfn(logits[mask], target[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        vrows, _ = eval_split(model, val, args.embedding_dir, device, args.min_sep, args.threshold)
        vF1 = agg(vrows, "F1"); vbF1 = agg(vrows, "best_F1"); vMCC = agg(vrows, "MCC")
        log_rows.append({"epoch": ep, "train_loss": tot / max(1, nb),
                         "val_F1": vF1, "val_best_F1": vbF1, "val_MCC": vMCC})
        val_rows_log.append({"epoch": ep, "val_F1": vF1, "val_best_F1": vbF1,
                             "val_MCC": vMCC, "val_AUPRC": agg(vrows, "AUPRC"),
                             "val_P_at_L": agg(vrows, "P_at_L")})
        print(f"  epoch {ep:2d}  loss={tot/max(1,nb):.4f}  val F1={vF1:.4f} best_F1={vbF1:.4f} MCC={vMCC:.4f}")
        if vF1 > best_f1:
            best_f1 = vF1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out / "best_model.pt")

    # test
    trows, preds = eval_split(model, test, args.embedding_dir, device, args.min_sep,
                              args.threshold, want_preds=True)
    test_recs = {r["id"]: r for r in test}
    for rid, pr in preds.items():
        np.save(out / "predictions" / f"{rid}.npy", pr.astype(np.float16))

    with open(out / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys())); w.writeheader(); w.writerows(log_rows)
    with open(out / "val_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows_log[0].keys())); w.writeheader(); w.writerows(val_rows_log)
    with open(out / "test_per_sequence.csv", "w", newline="") as f:
        if trows:
            w = csv.DictWriter(f, fieldnames=list(trows[0].keys())); w.writeheader(); w.writerows(trows)

    summary = {"num_train": len(train), "num_val": len(val), "num_test": len(trows),
               "contact_head_params": n_params, "pos_weight": pw,
               "best_val_F1": best_f1, "hidden_dim": hidden_dim}
    for key in ("precision", "recall", "F1", "best_F1", "MCC", "AUPRC", "AUROC",
                "P_at_L", "P_at_L2", "P_at_num_gold"):
        summary[f"mean_{key}"] = agg(trows, key, statistics.mean)
        summary[f"median_{key}"] = agg(trows, key, statistics.median)
        summary[f"std_{key}"] = agg(trows, key, statistics.pstdev) if len(trows) > 1 else 0.0
    json.dump(summary, open(out / "test_metrics.json", "w"), indent=2)

    # plots
    for n, (rid, pr) in enumerate(list(preds.items())[:args.num_plots]):
        r = test_recs[rid]; L = r["length"]
        G = np.zeros((L, L))
        for i, j in r["pairs"]:
            G[i, j] = G[j, i] = 1.0
        thr = (pr >= args.threshold).astype(float)
        mrow = next((x for x in trows if x["id"] == rid), {})
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(G, cmap="Greys"); ax[0].set_title("gold")
        im = ax[1].imshow(pr, cmap="viridis", vmin=0, vmax=1); ax[1].set_title("predicted prob")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        ax[2].imshow(G, cmap="Greys", alpha=0.4)
        yi, xi = np.where(np.triu(thr, args.min_sep) > 0)
        ax[2].scatter(xi, yi, s=8, c="red"); ax[2].scatter(yi, xi, s=8, c="red")
        ax[2].set_title(f"thresholded (>{args.threshold})")
        fig.suptitle(f"{rid} | {r['family']} | L={L} | F1={mrow.get('F1',0):.3f} "
                     f"MCC={mrow.get('MCC',0):.3f} AUPRC={mrow.get('AUPRC',0):.3f}", fontsize=11)
        fig.tight_layout(); fig.savefig(out / "example_plots" / f"{rid}.png", dpi=100); plt.close(fig)

    print(f"\n[test] n={len(trows)}  mean F1={summary['mean_F1']:.4f}  best_F1={summary['mean_best_F1']:.4f}  "
          f"MCC={summary['mean_MCC']:.4f}  AUPRC={summary['mean_AUPRC']:.4f}  P@L={summary['mean_P_at_L']:.4f}")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
