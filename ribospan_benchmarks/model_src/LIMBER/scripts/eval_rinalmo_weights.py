#!/usr/bin/env python3
"""Decisive pipeline validation: run RiNALMo's OWN released fine-tuned SS weights
through OUR decoder + metric, and check we reproduce their paper F1.

If we reproduce ~0.88 (5s) etc., our eval pipeline is proven correct and the gap
on our model is real (model/pretraining), NOT a pipeline bug.

Loads giga RiNALMo LM + SecStructPredictionHead from the fine-tuned .pt, runs
forward (representation -> strip CLS/EOS -> head -> sigmoid), then applies our
prob_mat_to_sec_struct + ss_f1 (already proven bit-identical to RiNALMo's).
"""
from __future__ import annotations
import argparse, statistics, sys
from pathlib import Path
import torch

sys.path.insert(0, "/tmp/RiNALMo")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rinalmo.config import model_config                       # noqa: E402
from rinalmo.model.model import RiNALMo                        # noqa: E402
from rinalmo.model.downstream import SecStructPredictionHead   # noqa: E402
from rinalmo.data.alphabet import Alphabet                     # noqa: E402
from rinalmo_ss_lib import parse_ct, prob_mat_to_sec_struct, ss_f1, ss_precision, ss_recall  # noqa: E402

THRESHOLDS = [i / 100 for i in range(1, 30)]


def load_split(d, max_len):
    items = []
    for ct in sorted(Path(d).glob("*.ct")):
        try:
            seq, pair = parse_ct(ct)
        except Exception:
            continue
        if 0 < len(seq) <= max_len:
            items.append((seq, pair))
    return items


@torch.no_grad()
def probs_for(lm, head, alpha, seq, device):
    ids = torch.tensor(alpha.batch_tokenize([seq])).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rep = lm(ids)["representation"]            # [1, L+2, d]
        logits = head(rep[..., 1:-1, :]).squeeze(-1)  # [1, L, L]
    return torch.sigmoid(logits.float())[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ct_root", default="data/contact_eval/raw/ct/fam-fold")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    device = args.device

    lm = RiNALMo(model_config("giga")).to(device).eval()
    head = SecStructPredictionHead(lm.config["model"]["transformer"].embed_dim, num_blocks=2).to(device).eval()
    sd = torch.load(args.weights, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    lm_sd = {k[len("lm."):]: v for k, v in sd.items() if k.startswith("lm.")}
    hd_sd = {k[len("pred_head."):]: v for k, v in sd.items() if k.startswith("pred_head.")}
    ml, ul = lm.load_state_dict(lm_sd, strict=False)
    mh, uh = head.load_state_dict(hd_sd, strict=False)
    print(f"[load] lm: {len(lm_sd)} keys (missing {len(ml)}, unexpected {len(ul)}) | "
          f"head: {len(hd_sd)} keys (missing {len(mh)}, unexpected {len(uh)})", flush=True)

    alpha = Alphabet()
    val = load_split(Path(args.ct_root) / args.family / "valid", args.max_len)
    test = load_split(Path(args.ct_root) / args.family / "test", args.max_len)
    print(f"[{args.family}] val/test = {len(val)}/{len(test)}", flush=True)

    # tune threshold on val (max mean flexible F1) -- same protocol as our pipeline
    per_t = {t: [] for t in THRESHOLDS}
    for seq, pair in val:
        p = probs_for(lm, head, alpha, seq, device)
        for t in THRESHOLDS:
            per_t[t].append(ss_f1(pair, prob_mat_to_sec_struct(p, seq, threshold=t)))
    meant = {t: statistics.mean(v) for t, v in per_t.items()}
    thr = max(meant, key=meant.get)
    print(f"[{args.family}] tuned threshold = {thr} (val F1 {meant[thr]:.3f})", flush=True)

    F, P, R = [], [], []
    for seq, pair in test:
        p = probs_for(lm, head, alpha, seq, device)
        ss = prob_mat_to_sec_struct(p, seq, threshold=thr)
        F.append(ss_f1(pair, ss)); P.append(ss_precision(pair, ss)); R.append(ss_recall(pair, ss))
    print(f"\n[RESULT] {args.family}: TEST F1={statistics.mean(F):.3f} "
          f"P={statistics.mean(P):.3f} R={statistics.mean(R):.3f} (n={len(test)}, thr={thr})", flush=True)


if __name__ == "__main__":
    main()
