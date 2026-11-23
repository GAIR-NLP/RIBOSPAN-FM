#!/usr/bin/env python3
"""Extract frozen LIMBER-650M CLS embeddings for the 27 RNA targets in ROBIN.
Modified-base notation like '(m6A)' is collapsed to the plain base (A) since
the kmer1 tokenizer only knows A/U/G/C -- fine for a first-pass baseline."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc  # noqa: E402
from _paths import PROJ

CKPT = str(PROJ / "checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12")
VOCAB_DIR = str(PROJECT_ROOT / "tokenizers" / "single")
SEQ_TABLE = PROJECT_ROOT / "data/robin/SMM_Sequence_Table_hit_rates.csv"
OUT_EMB = PROJECT_ROOT / "data/robin/rna_target_embeddings.npy"
OUT_META = PROJECT_ROOT / "data/robin/rna_target_meta.json"


def clean_seq(raw):
    # collapse "(m6A)" -> "A", "(mXY)" -> plain base Y as fallback, strip spaces/plus
    s = re.sub(r"\(m\d?([AUGC])\)", r"\1", raw)
    s = s.replace(" ", "").replace("+", "")
    return s.upper()


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cc.load_model(CKPT, "hybrid", device)
    tokenizer = cc.load_tokenizer(VOCAB_DIR)
    mm = cc.module_map(model, "hybrid")
    capture = {}
    mm["layers"][-1].register_forward_hook(
        lambda m, i, o: capture.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    rows = [r for r in csv.DictReader(open(SEQ_TABLE)) if r["Biomolecule Type"] == "RNA"]
    print(f"[robin_rna] {len(rows)} RNA targets", flush=True)

    embs, meta = [], []
    for r in rows:
        name = r["Nucleic Acid Target"]
        seq = clean_seq(r["Sequence (5' to 3')"])
        ids, am = cc.encode_kmer1(tokenizer, seq, device)
        model(input_ids=ids, attention_mask=am)
        h = capture["h"][0, 0]  # CLS token
        embs.append(h.float().cpu().numpy())
        meta.append({"target": name, "seq": seq, "structure_type": r["Structure Type"],
                     "hit_rate": float(r["Hit Rate"]), "selective_hit_rate": float(r["Selective Hit Rate"])})
        print(f"[robin_rna] {name:16s} len={len(seq):3d} {r['Structure Type']}", flush=True)

    embs = np.stack(embs, axis=0)
    np.save(OUT_EMB, embs)
    json.dump(meta, open(OUT_META, "w"), indent=2)
    print(f"[robin_rna] saved {embs.shape} -> {OUT_EMB}", flush=True)


if __name__ == "__main__":
    main()
