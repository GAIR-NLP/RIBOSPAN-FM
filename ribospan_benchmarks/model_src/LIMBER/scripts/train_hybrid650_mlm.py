#!/usr/bin/env python3
"""Hybrid-650M RNA foundation-model MLM pretraining, RiNALMo-Giga-aligned.

Backbone : hybrid Transformer+Mamba, 664M params (hidden 1280, 33 layers TTMx11,
           20 heads, FFN 5120, mamba_expand 4), SDPA attention. ctx 1022.
Data     : mmseqs-clustered RNAcentral, one randomly resampled member per
           cluster per epoch (scripts/sample_cluster_epochs.py), streamed from
           /dev/shm; on-the-fly random crop for L>1022. single-nuc tokenizer.
MLM      : 15% / 80-10-10, loss on masked only (scripts/rna_stream_dataset.py).
Optim    : AdamW lr 5e-5, wd 0.01, betas (0.9,0.999), eps 1e-8.
Sched    : warmup 2000 -> cosine decay to min_lr 1e-5.
Batch    : micro 8 x 8 gpu x accum 21 = effective 1344.
Budget   : 6 epochs. per-epoch checkpoint (rolling) + best-by-val-loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import (Trainer, TrainerCallback, TrainingArguments, set_seed)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa: E402
from rna_stream_dataset import (MLMCollator, RNAStreamDataset,  # noqa: E402
                                build_shm_cache_per_epoch, encode,
                                epoch_cache_is_fresh)
from _paths import PROJ  # noqa: E402


def build_val(val_path, max_len, n_max, seed):
    """Fixed, pre-masked validation set (deterministic) for comparable val loss /
    per-base PPL across epochs."""
    import random
    rng = random.Random(seed)
    coll = MLMCollator(seed=seed)
    feats = []
    with open(val_path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            ids, spec = encode(s, max_len, rng)        # deterministic given rng
            feats.append({"input_ids": ids, "special_tokens_mask": spec})
            if len(feats) >= n_max:
                break
    # pre-mask once with a fixed collator so eval is deterministic
    batches = [coll(feats[i:i + 64]) for i in range(0, len(feats), 64)]
    return batches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch_dir", default=str(PROJ / "data/rnacentral/epoch_seqs"),
                   help="dir with epoch{1..N}.txt + val.txt + meta.json from sample_cluster_epochs.py")
    p.add_argument("--shm_dir", default="/dev/shm")
    p.add_argument("--vocab_dir", default="tokenizers/single")
    p.add_argument("--output_dir", default="outputs/fm_hybrid_650m")
    p.add_argument("--max_len", type=int, default=1022)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=21)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--shuffle_buffer", type=int, default=50000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--val_steps", type=int, default=0, help="0 = once per epoch")
    p.add_argument("--save_steps", type=int, default=0,
                   help="checkpoint interval; 0 = once per epoch (fine for small "
                        "epoch_dir, but for full-data runs epoch boundaries can be "
                        "tens of thousands of steps apart -- pass an explicit small "
                        "value so a job-time-limit kill doesn't lose everything)")
    p.add_argument("--val_samples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--hidden_size", type=int, default=1280)
    p.add_argument("--num_attention_heads", type=int, default=20)
    p.add_argument("--intermediate_size", type=int, default=5120)
    p.add_argument("--layer_pattern", default="TTM" * 11)
    p.add_argument("--mamba_expand", type=int, default=4)
    p.add_argument("--mamba_d_state", type=int, default=16)
    p.add_argument("--mamba_d_conv", type=int, default=4)
    return p.parse_args()


class EpochCheckpointCallback(TrainerCallback):
    """Save a lightweight per-epoch model snapshot (weights only) that survives
    the rolling save_total_limit=1 checkpoint cleanup.  Written to
    <output_dir>/epoch_checkpoints/epoch{N}/  after each epoch end."""

    def __init__(self, output_dir, steps_per_epoch):
        self.out = Path(output_dir) / "epoch_checkpoints"
        self.spe = steps_per_epoch

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return
        epoch = round(state.global_step / self.spe)
        if epoch < 1:
            return
        import shutil
        ckpt_src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        epoch_dir = self.out / f"epoch{epoch}"
        if ckpt_src.exists() and not epoch_dir.exists():
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("pytorch_model.bin", "model.safetensors",
                          "config.json", "model_config.json"):
                src = ckpt_src / fname
                if src.exists():
                    shutil.copy2(src, epoch_dir / fname)
            # copy the model_config.json from output dir if not in checkpoint
            cfg_src = Path(args.output_dir) / "model_config.json"
            if cfg_src.exists() and not (epoch_dir / "model_config.json").exists():
                shutil.copy2(cfg_src, epoch_dir / "model_config.json")
            print(f"[epoch_ckpt] Saved epoch {epoch} -> {epoch_dir}", flush=True)


class CurveCallback(TrainerCallback):
    """Logs train loss to a history file + redraws the MLM-loss / per-base-PPL
    curve; computes fixed-val loss & PPL once per epoch."""

    def __init__(self, output_dir, model, val_batches, steps_per_epoch, val_steps):
        self.out = Path(output_dir)
        self.model = model
        self.val_batches = val_batches
        self.spe = steps_per_epoch
        self.val_steps = val_steps or steps_per_epoch
        self.hist = {"step": [], "train_loss": [], "lr": [],
                     "val_step": [], "val_loss": [], "val_ppl": [], "epoch_at_val": []}
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if not state.is_world_process_zero or not logs or "loss" not in logs:
            return
        self.hist["step"].append(state.global_step)
        self.hist["train_loss"].append(logs["loss"])
        self.hist["lr"].append(logs.get("learning_rate", 0.0))
        self._dump()

    def on_step_end(self, args, state, control, **kw):
        if state.global_step > 0 and state.global_step % self.val_steps == 0:
            self._validate(args, state)

    @torch.no_grad()
    def _validate(self, args, state):
        if not state.is_world_process_zero:
            return
        self.model.eval()
        dev = next(self.model.parameters()).device
        tot, ntok = 0.0, 0
        for b in self.val_batches:
            ids = b["input_ids"].to(dev); am = b["attention_mask"].to(dev); lab = b["labels"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=ids, attention_mask=am, labels=lab)
            n = int((lab != -100).sum())
            tot += out.loss.item() * n; ntok += n
        vl = tot / max(ntok, 1)
        self.hist["val_step"].append(state.global_step)
        self.hist["val_loss"].append(vl)
        self.hist["val_ppl"].append(math.exp(min(vl, 20)))
        self.hist["epoch_at_val"].append(state.global_step / self.spe)
        self.model.train()
        print(f"[val] step {state.global_step} (epoch {state.global_step/self.spe:.2f}) "
              f"val_loss={vl:.4f} per-base PPL={math.exp(min(vl,20)):.4f}", flush=True)
        self._dump(); self._plot()

    def _dump(self):
        json.dump(self.hist, open(self.out / "metrics_history.json", "w"))

    def _plot(self):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(13, 4))
            ax[0].plot(self.hist["step"], self.hist["train_loss"], lw=.7, label="train")
            if self.hist["val_step"]:
                ax[0].plot(self.hist["val_step"], self.hist["val_loss"], "o-", color="tab:red", label="val")
            ax[0].set_xlabel("step"); ax[0].set_ylabel("MLM loss"); ax[0].legend(); ax[0].grid(alpha=.3)
            if self.hist["val_step"]:
                ax[1].plot(self.hist["epoch_at_val"], self.hist["val_ppl"], "o-", color="tab:purple")
            ax[1].set_xlabel("epoch"); ax[1].set_ylabel("per-base PPL"); ax[1].grid(alpha=.3)
            fig.suptitle("Hybrid-650M MLM pretraining (full RNAcentral, ctx 1022)")
            fig.tight_layout(); fig.savefig(self.out / "training_curves.png", dpi=110); plt.close(fig)
        except Exception as e:
            print(f"[plot] {e}", flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    is_main = int(os.environ.get("RANK", "0")) == 0
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if is_main else (lambda *a: None)

    # ---- data cache (rank0 rebuilds if stale/missing; others just wait --
    # the purge+copy in build_shm_cache_per_epoch is not safe to run from
    # multiple ranks concurrently) ----
    meta = json.load(open(Path(args.epoch_dir) / "meta.json"))
    if is_main and not epoch_cache_is_fresh(args.epoch_dir, args.shm_dir, meta):
        build_shm_cache_per_epoch(args.epoch_dir, args.shm_dir, log=log)
    while not epoch_cache_is_fresh(args.epoch_dir, args.shm_dir, meta):
        time.sleep(10)
    n_train, val_shm, epoch_shm_paths = build_shm_cache_per_epoch(args.epoch_dir, args.shm_dir, log=log)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = math.ceil(n_train / (args.micro_batch * world_size * args.grad_accum))
    max_steps = steps_per_epoch * args.epochs
    log(f"[run] n_train={n_train:,}  steps/epoch={steps_per_epoch:,}  max_steps={max_steps:,}")

    # ---- model ----
    config = HybridMambaConfig(
        vocab_size=10, hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads, intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_len + 4, type_vocab_size=1,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, pad_token_id=0,
        layer_pattern=args.layer_pattern, mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv, mamba_expand=args.mamba_expand, tie_word_embeddings=False,
        attn_implementation="sdpa")
    model = HybridMambaForMaskedLM(config)
    if is_main:
        pc = model.param_groups()
        log(f"[model] params={pc['total']:,}  {pc}")
        json.dump(pc, open(out / "model_param_count.json", "w"), indent=2)
        json.dump(config.to_dict(), open(out / "model_config.json", "w"), indent=2, default=str)

    # Resuming past N already-trained epochs? Start the per-epoch cluster
    # resampling at file N instead of replaying epoch1's data again --
    # RNAStreamDataset's internal epoch counter otherwise always starts at 0,
    # independent of the Trainer's restored global_step.
    ckpts = list(Path(args.output_dir).glob("checkpoint-*"))
    start_epoch = 0
    if ckpts:
        state_f = ckpts[0] / "trainer_state.json"
        if state_f.exists():
            prev_step = json.load(open(state_f))["global_step"]
            start_epoch = prev_step // steps_per_epoch
            log(f"[resume] found checkpoint at step {prev_step} -> start_epoch={start_epoch}")

    train_ds = RNAStreamDataset(epoch_shm_paths, max_len=args.max_len,
                                shuffle_buffer=args.shuffle_buffer, seed=args.seed,
                                group_size=args.micro_batch, start_epoch=start_epoch)
    collator = MLMCollator(mlm_probability=args.mlm_probability, seed=args.seed)
    val_batches = build_val(val_shm, args.max_len, args.val_samples, args.seed) if is_main else []

    targs = TrainingArguments(
        output_dir=args.output_dir, overwrite_output_dir=False,
        do_train=True, do_eval=False, eval_strategy="no",
        max_steps=max_steps, save_strategy="steps",
        save_steps=args.save_steps or steps_per_epoch,
        save_total_limit=1, save_safetensors=False,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, max_grad_norm=1.0,
        lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr": args.min_lr},
        warmup_steps=args.warmup_steps, logging_steps=args.logging_steps,
        dataloader_num_workers=args.num_workers, dataloader_pin_memory=True,
        bf16=True, seed=args.seed, report_to="none", ddp_find_unused_parameters=False,
        accelerator_config={"dispatch_batches": False}, save_on_each_node=False,
        remove_unused_columns=False, ignore_data_skip=True)

    curve_cb = CurveCallback(args.output_dir, model, val_batches, steps_per_epoch, args.val_steps)
    epoch_ckpt_cb = EpochCheckpointCallback(args.output_dir, steps_per_epoch)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=collator, callbacks=[curve_cb, epoch_ckpt_cb])

    trainer.train(resume_from_checkpoint=bool(ckpts))
    trainer.save_model(args.output_dir)
    if is_main:
        json.dump({"n_train": n_train, "steps_per_epoch": steps_per_epoch,
                   "max_steps": max_steps, "tokens_per_epoch_approx": 18.85e9,
                   "effective_batch": args.micro_batch * 8 * args.grad_accum},
                  open(out / "run_budget.json", "w"), indent=2)
        # Marks the full run (all args.epochs) as actually finished -- lets the
        # SLURM self-requeue chain in pretrain_h200.sh stop submitting follow-ups.
        (out / "TRAINING_COMPLETE").write_text(f"max_steps={max_steps}\n")
        log(f"[done] saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
