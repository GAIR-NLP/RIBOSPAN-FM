"""Drop the 'optimizer' key from Zfold checkpoints: keeps the final evaluated model
(structure net + fine-tuned LM), removes training-only state. Atomic + verified."""
import glob, os, sys, torch
from _paths import PROJ

KEEP = ("epoch", "state_dict", "LM_state_dict")
total_before = total_after = 0
for f in sorted(glob.glob(str(PROJ / "rna3d/runs/*/checkpoints/model_ep*.pth"))):
    before = os.path.getsize(f)
    c = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    if "optimizer" not in c:
        print(f"skip (already stripped): {f}")
        total_before += before; total_after += before
        continue
    slim = {k: c[k] for k in KEEP if k in c}
    tmp = f + ".tmp"
    torch.save(slim, tmp)
    # verify the rewritten file before touching the original
    v = torch.load(tmp, map_location="cpu", weights_only=False, mmap=True)
    ok = (set(v) == set(slim)) and len(v.get("state_dict", {})) == len(c.get("state_dict", {})) \
         and len(v.get("LM_state_dict", {})) == len(c.get("LM_state_dict", {}))
    if not ok:
        os.remove(tmp)
        print(f"VERIFY FAILED, kept original: {f}", file=sys.stderr)
        total_before += before; total_after += before
        continue
    del c, v, slim
    os.replace(tmp, f)
    after = os.path.getsize(f)
    total_before += before; total_after += after
    print(f"{os.path.relpath(f, str(PROJ / 'rna3d/runs'))}: "
          f"{before/1e9:.2f} -> {after/1e9:.2f} GB")
print(f"\nTOTAL {total_before/1e9:.1f} GB -> {total_after/1e9:.1f} GB "
      f"(freed {(total_before-total_after)/1e9:.1f} GB)")
