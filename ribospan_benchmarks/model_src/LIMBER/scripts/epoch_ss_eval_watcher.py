#!/usr/bin/env python3
"""Per-epoch downstream evaluation watcher for the Hybrid-650M pretraining run.

Watches the training output dir for new epoch checkpoints. For each one it runs
the VALIDATED RiNALMo secondary-structure pipeline (RiNALMo head + canonical/
greedy decoder + flexible(+/-1) F1 + per-fold val-threshold tuning) on the
ArchiveII leave-one-family-out splits, and records family-wise F1 + macro F1 so
we get a per-epoch trajectory (Part 13/14).

Per-epoch eval uses the FROZEN-backbone probe (fast, light, shares GPUs with
training). The definitive gradual-unfreeze fine-tune comparison is run separately
on the final / best checkpoint (run_rinalmo_ss_ft_lfo.sh with the 650M model).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAMILIES = "5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA".split()
PYTHON = sys.executable


def latest_done_marker(out_dir):
    return out_dir / "ss_eval" / "evaluated_epochs.json"


def load_evaluated(out_dir):
    f = latest_done_marker(out_dir)
    return set(json.load(open(f))) if f.exists() else set()


def mark_evaluated(out_dir, epochs):
    f = latest_done_marker(out_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sorted(epochs), open(f, "w"))


def run(cmd, env=None, log=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run(cmd, env=e, stdout=log, stderr=subprocess.STDOUT)


def eval_epoch(out_dir, ckpt, epoch, embed_dim, log):
    """Frozen-probe RiNALMo SS LFO on one checkpoint -> family + macro F1."""
    tmp_model = Path(f"/dev/shm/h650_eval_ep{epoch}")
    cache = Path(f"/dev/shm/ss_cache_ep{epoch}")
    res_dir = out_dir / "ss_eval" / f"epoch{epoch}"
    res_dir.mkdir(parents=True, exist_ok=True)
    tmp_model.mkdir(parents=True, exist_ok=True)

    shutil.copy(out_dir / "model_config.json", tmp_model / "model_config.json")
    wsrc = ckpt / "pytorch_model.bin"
    if not wsrc.exists():
        print(f"[watcher] {ckpt} has no pytorch_model.bin yet, skip", flush=True)
        return None
    shutil.copy(wsrc, tmp_model / "pytorch_model.bin")
    for tk in ["vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
        src = PROJECT_ROOT / "tokenizers" / "single" / tk
        if src.exists():
            shutil.copy(src, tmp_model / tk)

    print(f"[watcher] epoch {epoch}: extracting frozen embeddings ...", flush=True)
    with open(res_dir / "extract.log", "w") as lg:
        r = run([PYTHON, "scripts/rinalmo_ss_extract.py", "--model_dir", str(tmp_model),
                 "--cache_dir", str(cache)], {"CUDA_VISIBLE_DEVICES": "0"}, lg)
    if r.returncode != 0:
        print(f"[watcher] epoch {epoch}: extract FAILED", flush=True)
        return None

    print(f"[watcher] epoch {epoch}: training 9 family heads ...", flush=True)
    procs = []
    for i, fam in enumerate(FAMILIES):
        lg = open(res_dir / f"{fam}.log", "w")
        e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"] = str(i % 8)
        procs.append(subprocess.Popen(
            [PYTHON, "scripts/rinalmo_ss_train.py", "--family", fam, "--cache_dir", str(cache),
             "--embed_dim", str(embed_dim), "--out_dir", str(res_dir)],
            env=e, stdout=lg, stderr=subprocess.STDOUT))
        if (i + 1) % 8 == 0:
            for p in procs:
                p.wait()
    for p in procs:
        p.wait()

    fam_f1 = {}
    for fam in FAMILIES:
        jf = res_dir / f"{fam}.json"
        if jf.exists():
            fam_f1[fam] = json.load(open(jf))["mean_F1"]
    macro = statistics.mean(fam_f1.values()) if fam_f1 else 0.0
    summary = {"epoch": epoch, "checkpoint_step": int(ckpt.name.split("-")[1]),
               "macro_F1": round(macro, 4), "family_F1": {k: round(v, 4) for k, v in fam_f1.items()},
               "method": "frozen-probe RiNALMo SS LFO (flexible F1)"}
    json.dump(summary, open(out_dir / "ss_eval" / f"epoch{epoch}.json", "w"), indent=2)
    print(f"[watcher] epoch {epoch}: macro_F1={macro:.4f}  {summary['family_F1']}", flush=True)

    shutil.rmtree(tmp_model, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)
    plot_curves(out_dir)
    return summary


def plot_curves(out_dir):
    sse = out_dir / "ss_eval"
    eps = sorted(int(p.stem[5:]) for p in sse.glob("epoch*.json") if p.stem[5:].isdigit())
    if not eps:
        return
    data = [json.load(open(sse / f"epoch{e}.json")) for e in eps]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(eps, [d["macro_F1"] for d in data], "o-", color="tab:blue")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("macro F1 (flexible)")
        ax[0].set_title("ArchiveII LFO macro F1 vs epoch"); ax[0].grid(alpha=.3); ax[0].set_ylim(0, 1)
        for fam in FAMILIES:
            ys = [d["family_F1"].get(fam) for d in data]
            ax[1].plot(eps, ys, "o-", label=fam, lw=1)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("F1"); ax[1].set_ylim(0, 1)
        ax[1].set_title("family-wise F1 vs epoch"); ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=.3)
        fig.suptitle("Hybrid-650M downstream (RiNALMo SS pipeline, frozen probe)")
        fig.tight_layout(); fig.savefig(out_dir / "ss_eval_curves.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[watcher] plot error: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/fm_hybrid_650m")
    ap.add_argument("--steps_per_epoch", type=int, default=30278)
    ap.add_argument("--embed_dim", type=int, default=1280)
    ap.add_argument("--poll_sec", type=int, default=300)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    print(f"[watcher] watching {out_dir} (spe={args.steps_per_epoch}); polling every {args.poll_sec}s", flush=True)

    while True:
        done = load_evaluated(out_dir)
        ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        for ckpt in ckpts:
            step = int(ckpt.name.split("-")[1])
            epoch = round(step / args.steps_per_epoch)
            if epoch < 1 or epoch in done:
                continue
            if abs(step - epoch * args.steps_per_epoch) > 0.2 * args.steps_per_epoch:
                continue
            print(f"[watcher] new epoch {epoch} checkpoint at step {step}", flush=True)
            if eval_epoch(out_dir, ckpt, epoch, args.embed_dim, None) is not None:
                done.add(epoch); mark_evaluated(out_dir, done)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
