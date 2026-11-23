#!/usr/bin/env python3
"""MLM pretraining with the Transformer+Mamba hybrid backbone.

Reuses train_bert_mlm.py's streaming data pipeline, MLM collator, live-chart
callback, and per-batch metrics; ONLY the model is swapped from BertForMaskedLM
to HybridMambaForMaskedLM. All training hyperparameters default to the kmer1
BERT baseline so the comparison is backbone-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
# reuse the baseline's callback + metric helpers verbatim
from train_bert_mlm import (  # noqa: E402
    LiveChartCallback,
    compute_metrics,
    preprocess_logits_for_metrics,
)
from models.hybrid_mamba_bert import (  # noqa: E402
    HybridMambaConfig,
    HybridMambaForMaskedLM,
)


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Transformer+Mamba MLM pretraining.")
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
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=54231)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_eval_samples", type=int, default=2000)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # hybrid backbone (match BERT baseline dims; only backbone differs)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1024)
    p.add_argument("--layer_pattern", default="TTMTTM")
    p.add_argument("--mamba_d_state", type=int, default=16)
    p.add_argument("--mamba_d_conv", type=int, default=4)
    p.add_argument("--mamba_expand", type=int, default=4)
    p.add_argument("--tie_word_embeddings", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    is_main = int(os.environ.get("RANK", "0")) == 0

    vocab_file = os.path.join(args.vocab_dir, "vocab.txt")
    tokenizer = BertTokenizerFast(
        vocab_file=vocab_file, do_lower_case=False, do_basic_tokenize=True,
        tokenize_chinese_chars=False, strip_accents=False)
    vocab_size = tokenizer.vocab_size

    config = HybridMambaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_seq_length + 2, 514),
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        layer_pattern=args.layer_pattern,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        tie_word_embeddings=args.tie_word_embeddings,
    )
    model = HybridMambaForMaskedLM(config)
    pcounts = model.param_groups()
    if is_main:
        print(f"[hybrid] vocab_size  : {vocab_size}")
        print(f"[hybrid] pattern     : {args.layer_pattern}")
        print(f"[hybrid] params      : {pcounts['total']:,}  {pcounts}")

    # ----- data (same streaming pipeline as the BERT baseline) -----
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length)

    if args.streaming:
        train_dataset = load_dataset(
            "text", data_files={"train": args.train_file}, split="train",
            streaming=True).shuffle(seed=args.seed, buffer_size=20000).map(
            tokenize_fn, batched=True, remove_columns=["text"])
        val_raw = load_dataset(
            "text", data_files={"validation": args.validation_file},
            split="validation")
        eval_dataset = val_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    else:
        raw = load_dataset("text", data_files={
            "train": args.train_file, "validation": args.validation_file})
        tok = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
        train_dataset, eval_dataset = tok["train"], tok["validation"]

    if args.max_eval_samples and len(eval_dataset) > args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
        pad_to_multiple_of=args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir, overwrite_output_dir=True,
        do_train=True, do_eval=True, eval_strategy="steps",
        eval_steps=args.eval_steps, save_steps=args.save_steps, save_total_limit=3,
        logging_steps=args.logging_steps, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        seed=args.seed, report_to="none", ddp_find_unused_parameters=False,
        fp16=args.fp16, bf16=args.bf16,
        accelerator_config={"dispatch_batches": False},
        # BertOnlyMLMHead ties predictions.bias and decoder.bias (shared memory);
        # safetensors rejects shared tensors, so save as pytorch_model.bin.
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator, compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[LiveChartCallback(args.output_dir, "hybrid_mamba")],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    import math
    if "eval_loss" in metrics:
        try:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["eval_perplexity"] = float("inf")

    if trainer.is_world_process_zero():
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=2, default=str)
        with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        with open(os.path.join(args.output_dir, "model_param_count.json"), "w") as f:
            json.dump(pcounts, f, indent=2)
        print(f"[hybrid] final eval: {metrics}")
        print(f"[hybrid] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
