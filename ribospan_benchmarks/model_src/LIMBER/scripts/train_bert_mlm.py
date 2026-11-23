#!/usr/bin/env python3
"""Plain BERT masked-language-model pretraining for RNA sequences.

Compares a single-base tokenizer vs a centered-3-mer tokenizer. The model
architecture is identical in both cases; only the tokenizer / vocab differs.

This is vanilla MLM: BertForMaskedLM + DataCollatorForLanguageModeling
(mlm_probability=0.15). No structure, pairing bias, contrastive loss, span
masking, or any custom loss.

Works single-GPU or multi-GPU via torchrun (HF Trainer handles DDP):

    torchrun --nproc_per_node=8 scripts/train_bert_mlm.py --tokenizer_type single ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: render charts to PNG without a display
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


class LiveChartCallback(TrainerCallback):
    """Record train/eval metrics and (re)render a live PNG chart + CSV/JSON.

    Updates every time the trainer logs a train loss (--logging_steps) or runs
    an evaluation (--eval_steps), so you can watch convergence in real time.
    Only the main process writes files.
    """

    def __init__(self, output_dir: str, tokenizer_type: str):
        self.output_dir = output_dir
        self.tag = tokenizer_type
        self.train_steps: list[int] = []
        self.train_loss: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_loss: list[float] = []
        self.eval_acc: list[float] = []
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        # train-loss log lines have 'loss'; eval lines have 'eval_loss'
        if "loss" in logs and "eval_loss" not in logs:
            self.train_steps.append(state.global_step)
            self.train_loss.append(float(logs["loss"]))
            self._flush()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or not metrics:
            return
        self.eval_steps.append(state.global_step)
        self.eval_loss.append(float(metrics.get("eval_loss", float("nan"))))
        self.eval_acc.append(float(metrics.get("eval_masked_accuracy", float("nan"))))
        self._flush()

    def _flush(self):
        # JSON history (full series)
        hist = {
            "tokenizer_type": self.tag,
            "train": {"step": self.train_steps, "loss": self.train_loss},
            "eval": {
                "step": self.eval_steps,
                "loss": self.eval_loss,
                "masked_accuracy": self.eval_acc,
            },
        }
        with open(os.path.join(self.output_dir, "metrics_history.json"), "w") as fh:
            json.dump(hist, fh, indent=2)
        # CSV of eval points (easy to tail / watch)
        with open(os.path.join(self.output_dir, "eval_metrics.csv"), "w") as fh:
            fh.write("step,eval_loss,masked_accuracy\n")
            for s, l, a in zip(self.eval_steps, self.eval_loss, self.eval_acc):
                fh.write(f"{s},{l:.6f},{a:.6f}\n")
        self._plot()

    def _plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax0, ax1 = axes
        if self.train_steps:
            ax0.plot(self.train_steps, self.train_loss,
                     color="tab:blue", alpha=0.6, label="train loss")
        if self.eval_steps:
            ax0.plot(self.eval_steps, self.eval_loss,
                     color="tab:red", marker="o", label="eval loss")
        ax0.set_xlabel("step")
        ax0.set_ylabel("MLM loss")
        ax0.set_title(f"[{self.tag}] loss")
        ax0.legend()
        ax0.grid(True, alpha=0.3)

        if self.eval_steps:
            ax1.plot(self.eval_steps, [a * 100 for a in self.eval_acc],
                     color="tab:green", marker="o", label="masked-token accuracy")
            if self.eval_acc:
                last = self.eval_acc[-1] * 100
                ax1.annotate(f"{last:.1f}%",
                             xy=(self.eval_steps[-1], last),
                             xytext=(4, 4), textcoords="offset points")
        ax1.set_xlabel("step")
        ax1.set_ylabel("masked-token accuracy (%)")
        ax1.set_title(f"[{self.tag}] masked-token prediction success rate")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "live_metrics.png"), dpi=110)
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="BERT MLM pretraining for RNA.")
    p.add_argument("--tokenizer_type", required=True,
                   help="Label for the run (e.g. single, kmer3, kmer4). "
                        "Vocabulary is read from --vocab_dir; this only names "
                        "outputs / chart titles.")
    p.add_argument("--train_file", required=True)
    p.add_argument("--validation_file", required=True)
    p.add_argument("--vocab_dir", required=True,
                   help="Directory containing vocab.txt for this tokenizer.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_eval_samples", type=int, default=2000,
                   help="Cap the validation set to this many sequences for a "
                        "fast, frequent live eval (0 = use all).")
    p.add_argument("--streaming", action="store_true",
                   help="Stream + tokenize the train file on the fly (for large "
                        "foundation-model data; avoids upfront tokenization).")
    # Model size (identical across tokenizers; only vocab differs).
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1024)
    return p.parse_args()


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax predictions so eval doesn't store full vocab
    distributions across the whole eval set (saves a lot of memory)."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Masked-token prediction success rate: of the tokens that were masked
    out (label != -100), the fraction the model predicts correctly."""
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    mask = labels != -100
    n = int(mask.sum())
    if n == 0:
        return {"masked_accuracy": 0.0, "masked_token_count": 0}
    correct = int((preds[mask] == labels[mask]).sum())
    return {"masked_accuracy": correct / n, "masked_token_count": n}


def main():
    args = parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------ #
    # Tokenizer (whitespace-pre-split tokens -> direct vocab lookup)
    # ------------------------------------------------------------------ #
    vocab_file = os.path.join(args.vocab_dir, "vocab.txt")
    if not os.path.exists(vocab_file):
        raise SystemExit(f"[error] vocab.txt not found in {args.vocab_dir}")
    tokenizer = BertTokenizerFast(
        vocab_file=vocab_file,
        do_lower_case=False,
        do_basic_tokenize=True,
        tokenize_chinese_chars=False,
        strip_accents=False,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    vocab_size = tokenizer.vocab_size
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        print(f"[train] tokenizer_type : {args.tokenizer_type}")
        print(f"[train] vocab_size      : {vocab_size}")

    # ------------------------------------------------------------------ #
    # Model config (identical across tokenizers, only vocab_size differs)
    # ------------------------------------------------------------------ #
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_seq_length + 2, 514),
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] model params    : {n_params:,}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
        )

    if args.streaming:
        # Large foundation-model data: stream the train file and tokenize
        # lazily in the dataloader workers (no multi-GB upfront tokenize, no
        # tokenized-arrow storage). Validation stays a small in-memory set.
        train_stream = load_dataset(
            "text", data_files={"train": args.train_file},
            split="train", streaming=True,
        ).shuffle(seed=args.seed, buffer_size=20000)
        train_dataset = train_stream.map(
            tokenize_fn, batched=True, remove_columns=["text"])
        val_raw = load_dataset(
            "text", data_files={"validation": args.validation_file},
            split="validation")
        eval_dataset = val_raw.map(
            tokenize_fn, batched=True, remove_columns=["text"], desc="tokenizing val")
    else:
        raw = load_dataset(
            "text",
            data_files={"train": args.train_file, "validation": args.validation_file},
        )
        tokenized = raw.map(
            tokenize_fn, batched=True, remove_columns=["text"], desc="tokenizing")
        train_dataset = tokenized["train"]
        eval_dataset = tokenized["validation"]

    if args.max_eval_samples and len(eval_dataset) > args.max_eval_samples:
        # Small, fixed validation subset for fast, frequent live monitoring.
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if is_main:
            print(f"[train] eval subset     : {len(eval_dataset)} sequences "
                  f"(capped by --max_eval_samples)")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        # Pad every batch to exactly max_seq_length so all ranks/batches share a
        # shape — required for DDP + streaming IterableDataset and for gathering
        # eval predictions across ranks (dynamic padding mismatches otherwise).
        pad_to_multiple_of=args.max_seq_length,
    )

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        report_to="none",
        ddp_find_unused_parameters=False,
        fp16=True,
        # Required for DDP + streaming IterableDataset: each process fetches its
        # own (variable-size) batch instead of the main process dispatching one.
        accelerator_config={"dispatch_batches": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[LiveChartCallback(args.output_dir, args.tokenizer_type)],
    )

    trainer.train()

    # ------------------------------------------------------------------ #
    # Save final model + artifacts (main process only)
    # ------------------------------------------------------------------ #
    trainer.save_model(args.output_dir)            # final model
    trainer.save_state()                           # trainer_state.json
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        import math
        try:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["eval_perplexity"] = float("inf")

    if trainer.is_world_process_zero():
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as fh:
            json.dump(training_args.to_dict(), fh, indent=2, default=str)
        print(f"[train] final eval: {metrics}")
        print(f"[train] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
