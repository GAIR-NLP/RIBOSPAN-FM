#!/usr/bin/env python3
"""Decoder-only (GPT-2 style) causal-LM pretraining: next-token prediction.

Same tokenizer / data / streaming pipeline / batch / optimizer as the MLM
baselines, sized to match (n_embd 256, n_layer 6, n_head 4, n_inner 1024 ->
~4.87M kmer1 / ~8.76M kmer6, within ±10% of the BERT baselines). The OBJECTIVE
differs: autoregressive next-token prediction (causal mask), not MLM.

Metrics: next-token accuracy (shifted) and per-token / per-base perplexity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_bert_mlm import LiveChartCallback  # noqa: E402  (reuse live chart)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Shifted next-token accuracy: position t predicts token t+1."""
    preds, labels = eval_pred
    preds = np.asarray(preds)[:, :-1]
    labels = np.asarray(labels)[:, 1:]
    mask = labels != -100
    n = int(mask.sum())
    if n == 0:
        return {"masked_accuracy": 0.0, "next_token_count": 0}
    correct = int((preds[mask] == labels[mask]).sum())
    # key name "masked_accuracy" so the shared LiveChartCallback plots it
    return {"masked_accuracy": correct / n, "next_token_count": n}


def parse_args():
    p = argparse.ArgumentParser(description="Decoder-only (GPT2) causal LM pretraining.")
    p.add_argument("--tokenizer_type", default="single")
    p.add_argument("--train_file", required=True)
    p.add_argument("--validation_file", required=True)
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=128)
    p.add_argument("--per_device_eval_batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=54231)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_eval_samples", type=int, default=2000)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_inner", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    is_main = int(os.environ.get("RANK", "0")) == 0

    tokenizer = BertTokenizerFast(
        vocab_file=os.path.join(args.vocab_dir, "vocab.txt"),
        do_lower_case=False, do_basic_tokenize=True,
        tokenize_chinese_chars=False, strip_accents=False)
    vocab_size = tokenizer.vocab_size

    config = GPT2Config(
        vocab_size=vocab_size, n_positions=args.max_seq_length,
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        n_inner=args.n_inner, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id)
    model = GPT2LMHeadModel(config)
    if is_main:
        n = sum(p.numel() for p in model.parameters())
        print(f"[decoder] tokenizer={args.tokenizer_type} vocab={vocab_size} params={n:,}")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length)

    if args.streaming:
        train_dataset = load_dataset(
            "text", data_files={"train": args.train_file}, split="train",
            streaming=True).shuffle(seed=args.seed, buffer_size=20000).map(
            tokenize_fn, batched=True, remove_columns=["text"])
        val_raw = load_dataset("text", data_files={"validation": args.validation_file},
                               split="validation")
        eval_dataset = val_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    else:
        raw = load_dataset("text", data_files={
            "train": args.train_file, "validation": args.validation_file})
        tok = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
        train_dataset, eval_dataset = tok["train"], tok["validation"]

    if args.max_eval_samples and len(eval_dataset) > args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    # mlm=False => causal LM collator: labels = input_ids, pad labels set to -100.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir, overwrite_output_dir=True,
        do_train=True, do_eval=True, eval_strategy="steps",
        eval_steps=args.eval_steps, save_steps=args.save_steps, save_total_limit=3,
        logging_steps=args.logging_steps, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, dataloader_num_workers=args.num_workers,
        seed=args.seed, report_to="none", ddp_find_unused_parameters=False,
        fp16=args.fp16, bf16=args.bf16,
        accelerator_config={"dispatch_batches": False})

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator, compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[LiveChartCallback(args.output_dir, f"decoder_{args.tokenizer_type}")])

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["eval_perplexity"] = float("inf")
    if trainer.is_world_process_zero():
        json.dump(metrics, open(os.path.join(args.output_dir, "eval_results.json"), "w"), indent=2)
        json.dump(training_args.to_dict(), open(os.path.join(args.output_dir, "training_args.json"), "w"), indent=2, default=str)
        json.dump(config.to_dict(), open(os.path.join(args.output_dir, "model_config.json"), "w"), indent=2, default=str)
        print(f"[decoder] final eval: {metrics}")
        print(f"[decoder] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
