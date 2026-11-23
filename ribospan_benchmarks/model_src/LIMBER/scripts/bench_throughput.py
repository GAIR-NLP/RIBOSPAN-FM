#!/usr/bin/env python3
"""Micro-benchmark fwd+bwd throughput for Hybrid-650M on H200.

Tests micro_batch sizes [16, 32, 48, 64] at ctx=1022. Picks largest that fits.
Then prints the required grad_accum to maintain effective batch = 1344 with 8 GPUs.

Run: python scripts/bench_throughput.py
"""
import sys, time, json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM  # noqa

DEVICE = "cuda:0"
CTX = 1022
VOCAB_SIZE = 10
TARGET_EFF_BATCH = 1344
N_GPUS = 4
MICRO_BATCHES = [16, 32, 48, 64, 96, 128, 168]
WARMUP = 2
MEASURE = 5


def build_model():
    cfg = HybridMambaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=1280,
        num_hidden_layers=33,
        num_attention_heads=20,
        intermediate_size=5120,
        max_position_embeddings=1026,
        layer_pattern="TTMTTMTTMTTMTTMTTMTTMTTMTTMTTMTTM",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=4,
    )
    model = HybridMambaForMaskedLM(cfg).to(DEVICE)
    model.train()
    return model


def bench_one(model, micro_batch):
    ids = torch.randint(5, VOCAB_SIZE, (micro_batch, CTX), device=DEVICE)
    labels = ids.clone()
    labels[labels != 4] = -100  # only 'masked' positions contribute to loss
    labels[:, :CTX // 7] = ids[:, :CTX // 7]  # ~15% masked

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # warm up
    for _ in range(WARMUP):
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    seqs_per_sec = micro_batch * MEASURE / elapsed
    return seqs_per_sec


def main():
    print(f"[bench] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[bench] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[bench] Building Hybrid-664M model...")
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] Params: {n_params/1e6:.1f}M")
    print()

    results = {}
    best_mb, best_thr = 0, 0.0
    for mb in MICRO_BATCHES:
        try:
            torch.cuda.empty_cache()
            thr = bench_one(model, mb)
            mem = torch.cuda.max_memory_allocated() / 1e9
            # grad_accum to hit effective batch = 1344 with N_GPUS
            eff = mb * N_GPUS
            grad_accum = TARGET_EFF_BATCH // eff
            remainder = TARGET_EFF_BATCH % eff
            print(f"  micro_batch={mb:3d}  {thr:.1f} seqs/s  mem={mem:.1f}GB"
                  f"  -> grad_accum={grad_accum} (eff={eff*grad_accum}"
                  + (f", remainder={remainder}" if remainder else "") + ")")
            results[mb] = {"throughput": thr, "mem_gb": mem, "grad_accum": grad_accum}
            if grad_accum >= 1:
                best_mb, best_thr = mb, thr
        except torch.cuda.OutOfMemoryError:
            print(f"  micro_batch={mb:3d}  OOM")
            torch.cuda.empty_cache()
            break

    print(f"\n[bench] RECOMMENDATION: micro_batch={best_mb}, grad_accum={TARGET_EFF_BATCH//(best_mb*N_GPUS)}")
    print(f"  Effective batch = {best_mb * N_GPUS * (TARGET_EFF_BATCH//(best_mb*N_GPUS))}")

    out = PROJECT_ROOT / "outputs" / "fm_hybrid_650m" / "h200_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "best_micro_batch": best_mb,
               "best_grad_accum": TARGET_EFF_BATCH // (best_mb * N_GPUS)}, open(out, "w"), indent=2)
    print(f"[bench] Saved: {out}")


if __name__ == "__main__":
    main()
