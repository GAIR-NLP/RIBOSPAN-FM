#!/usr/bin/env python3
"""Masked-LM perplexity on the standard 4096-seq val slice and on the subset of
it that a given model never saw during pretraining.

    python ppl_clean_val.py --ckpt <dir> --train_file <epoch1.txt> --tag 20pct
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layer_importance import VAL_PATH, MAX_LEN, SEED, build_val_batches, run_batches  # noqa
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa
from _paths import PROJ

CFG_KEYS = ("vocab_size", "hidden_size", "num_attention_heads", "intermediate_size",
            "max_position_embeddings", "type_vocab_size", "hidden_dropout_prob",
            "attention_probs_dropout_prob", "pad_token_id", "layer_pattern",
            "mamba_d_state", "mamba_d_conv", "mamba_expand", "tie_word_embeddings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_val", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(PROJ / "outputs/val_leak"))
    a = ap.parse_args()

    # the exact 4096 sequences the reported perplexity is computed on
    slice_seqs = []
    with open(VAL_PATH) as fh:
        for line in fh:
            s = line.strip()
            if s:
                slice_seqs.append(s)
            if len(slice_seqs) >= a.n_val:
                break

    print(f"[{a.tag}] val slice: {len(slice_seqs)} seqs; scanning training file ...", flush=True)
    wanted = set(slice_seqs)
    seen = set()
    with open(a.train_file) as fh:
        for line in fh:
            s = line.strip()
            if s in wanted:
                seen.add(s)
    clean = [s for s in slice_seqs if s not in seen]
    print(f"[{a.tag}] leaked {len(seen)} / {len(slice_seqs)}  -> clean {len(clean)}", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clean_path = out / f"val_clean_{a.tag}.txt"
    clean_path.write_text("".join(s + "\n" for s in clean))

    cfg = json.load(open(Path(a.ckpt) / "model_config.json"))
    model = HybridMambaForMaskedLM(HybridMambaConfig(**{k: cfg[k] for k in CFG_KEYS if k in cfg}))
    sd = torch.load(Path(a.ckpt) / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda").eval()

    res = {"tag": a.tag, "ckpt": a.ckpt, "n_slice": len(slice_seqs),
           "n_leaked": len(seen), "n_clean": len(clean)}
    for name, path, n in (("reported", VAL_PATH, a.n_val),
                          ("clean", str(clean_path), len(clean))):
        batches = build_val_batches(path, MAX_LEN, n, SEED, a.batch)
        ppl, _, _ = run_batches(model, batches, "cuda", "full", 0)
        res[f"ppl_{name}"] = float(ppl)
        print(f"[{a.tag}] ppl_{name} = {ppl:.4f}  ({n} seqs)", flush=True)
    res["delta"] = res["ppl_clean"] - res["ppl_reported"]
    print(f"[{a.tag}] RESULT {json.dumps(res)}", flush=True)
    (out / f"{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
