import sys, time, torch
from _paths import REPO
sys.path.insert(0, str(REPO))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM

DEVICE="cuda:0"; CTX=1022; VOCAB_SIZE=10
cfg = HybridMambaConfig(vocab_size=VOCAB_SIZE, hidden_size=1280, num_hidden_layers=33,
    num_attention_heads=20, intermediate_size=5120, max_position_embeddings=1026,
    layer_pattern="TTMTTMTTMTTMTTMTTMTTMTTMTTMTTMTTM", mamba_d_state=16, mamba_d_conv=4, mamba_expand=4)
model = HybridMambaForMaskedLM(cfg).to(DEVICE); model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
for mb in [4, 8, 12]:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(5, VOCAB_SIZE, (mb, CTX), device=DEVICE)
        labels = ids.clone(); labels[labels != 4] = -100; labels[:, :CTX//7] = ids[:, :CTX//7]
        for _ in range(2):
            out = model(input_ids=ids, labels=labels); out.loss.backward(); opt.zero_grad()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(5):
            out = model(input_ids=ids, labels=labels); out.loss.backward(); opt.zero_grad()
        torch.cuda.synchronize(); el=time.perf_counter()-t0
        mem = torch.cuda.max_memory_allocated()/1e9
        print(f"  micro_batch={mb}  {mb*5/el:.1f} seqs/s  mem={mem:.1f}GB")
    except torch.cuda.OutOfMemoryError:
        print(f"  micro_batch={mb}  OOM"); torch.cuda.empty_cache()
