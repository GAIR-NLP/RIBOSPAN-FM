"""Faithful reimplementation of RiNALMo's secondary-structure pipeline
components (head, decoder, metric, CT parsing), so we can run our hybrid-300M
backbone through the SAME downstream pipeline. RiNALMo is MIT-licensed; this is
an independent reimplementation matching their official behaviour exactly:

  - SecStructPredictionHead: outer-concat pair rep (2*d) -> Linear(2d->64) ->
    `num_blocks` bottleneck ResNet2D (1x1 / 3x3-same / 1x1, InstanceNorm, ReLU,
    residual) -> Conv2d(64->1, 3x3) -> symmetrize (triu(k=1) + transpose).
  - prob_mat_to_sec_struct: mask non-canonical pairs + sharp loops (|i-j|<4),
    threshold, then greedy max-prob matching (each base pairs at most once).
  - ss_precision/recall/f1: FLEXIBLE (+/-1 shift) evaluation on the upper
    triangle (k=1), via sklearn precision/recall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score

VALID = set("AUCGN")
CANONICAL_PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}
SHARP_LOOP_DIST = 4


# --------------------------------------------------------------------------- #
# CT parsing (RiNALMo's fam-fold ArchiveII files)
# --------------------------------------------------------------------------- #
def parse_ct(path: Path):
    lines = [ln for ln in Path(path).read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    n = int(lines[0].split()[0])
    pair = np.zeros((n, n), dtype=np.float32)
    seq = []
    for ln in lines[1:1 + n]:
        p = ln.split()
        i, nuc, j = int(p[0]), p[1], int(p[4])
        seq.append(nuc)
        if j > 0:
            pair[i - 1, j - 1] = 1.0
    s = "".join(seq).upper().replace("T", "U")
    s = "".join(c if c in VALID else "N" for c in s)
    return s, pair


# --------------------------------------------------------------------------- #
# Head (exact RiNALMo architecture)
# --------------------------------------------------------------------------- #
def _outer_concat(t):  # [B,L,d] -> [B,L,L,2d]
    L = t.shape[1]
    a = t.unsqueeze(-2).expand(-1, -1, L, -1)
    b = t.unsqueeze(-3).expand(-1, L, -1, -1)
    return torch.cat((a, b), dim=-1)


class _ResNet2DBlock(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False), nn.InstanceNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, k, bias=False, padding="same"), nn.InstanceNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False), nn.InstanceNorm2d(c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.net(x)


class SecStructPredictionHead(nn.Module):
    def __init__(self, embed_dim, num_blocks=2, conv_dim=64, kernel_size=3):
        super().__init__()
        self.linear_in = nn.Linear(embed_dim * 2, conv_dim)
        self.resnet = nn.ModuleList([_ResNet2DBlock(conv_dim, kernel_size) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size, padding="same")

    def forward(self, x):                 # x: [B,L,d]
        x = self.linear_in(_outer_concat(x))      # [B,L,L,64]
        x = x.permute(0, 3, 1, 2)                 # [B,64,L,L]
        for blk in self.resnet:
            x = blk(x)
        x = self.conv_out(x).squeeze(-3)          # [B,L,L]
        x = torch.triu(x, diagonal=1)
        return x + x.transpose(-1, -2)            # symmetric


# --------------------------------------------------------------------------- #
# Decoder (exact RiNALMo post-processing)
# --------------------------------------------------------------------------- #
def _sharp_loop_mask(L):
    m = np.eye(L, dtype=bool)
    for k in range(1, SHARP_LOOP_DIST):
        m |= np.eye(L, k=k, dtype=bool) | np.eye(L, k=-k, dtype=bool)
    return m


_BASE_IDX = {"A": 0, "U": 1, "C": 2, "G": 3, "N": 4}
_PAIR_OK = np.zeros((5, 5), dtype=bool)
for _a, _b in CANONICAL_PAIRS:
    _PAIR_OK[_BASE_IDX[_a], _BASE_IDX[_b]] = True


def _canonical_mask(seq):
    # vectorized equivalent of the O(L^2) python double-loop over CANONICAL_PAIRS
    idx = np.fromiter((_BASE_IDX.get(c, 4) for c in seq), dtype=np.intp, count=len(seq))
    return _PAIR_OK[idx[:, None], idx[None, :]]


def _greedy_clean(ss, probs):
    clean = np.copy(ss)
    tmp = np.copy(probs)
    tmp[ss < 1] = 0.0
    while np.sum(tmp > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(tmp), tmp.shape)
        tmp[i, :] = tmp[j, :] = tmp[:, i] = tmp[:, j] = 0.0
        clean[i, :] = clean[j, :] = clean[:, i] = clean[:, j] = 0
        clean[i, j] = clean[j, i] = 1
    return clean


def prob_mat_to_sec_struct(probs, seq, threshold=0.5,
                           allow_nc=False, allow_sharp=False):
    probs = np.array(probs, dtype=np.float64)
    L = probs.shape[-1]
    allowed = ~np.eye(L, dtype=bool)
    if not allow_sharp:
        allowed &= ~_sharp_loop_mask(L)
    if not allow_nc:
        allowed &= _canonical_mask(seq)
    probs = probs.copy()
    probs[~allowed] = 0.0
    ss = (probs > threshold).astype(int)
    return _greedy_clean(ss, probs)


# --------------------------------------------------------------------------- #
# Metric (exact RiNALMo flexible +/-1 F1)
# --------------------------------------------------------------------------- #
def _relax(ss):
    ss = np.pad(ss, ((1, 1), (1, 1)))
    relax = (np.roll(ss, 1, -1) + np.roll(ss, -1, -1)
             + np.roll(ss, 1, -2) + np.roll(ss, -1, -2))
    out = np.clip(ss + relax, 0, 1)[..., 1:-1, 1:-1]
    return out


def ss_precision(target, pred, flexible=True):
    if flexible:
        target = _relax(target)
    idx = np.triu_indices(target.shape[-1], k=1)
    return precision_score(target[idx], pred[idx], zero_division=0.0)


def ss_recall(target, pred, flexible=True):
    if flexible:
        pred = _relax(pred)
    idx = np.triu_indices(target.shape[-1], k=1)
    return recall_score(target[idx], pred[idx], zero_division=0.0)


def ss_f1(target, pred, flexible=True):
    p = ss_precision(target, pred, flexible)
    r = ss_recall(target, pred, flexible)
    return 0.0 if p + r < 1e-5 else 2 * p * r / (p + r)


# ---- Standard SS benchmark parsers (bit-faithful to RiNALMo's rinalmo/utils/sec_struct.py) ----
# .ct  : ArchiveII fam-fold splits
# .st  : SPOT-RNA bpRNA dataset (TR0 / VL0 / TS0)
# .bpseq: Rivas TrainSetA / TestSetA / TestSetB

_CLOSE_TO_OPEN = {")": "(", "]": "[", "}": "{", ">": "<"}


def normalize_seq(seq):
    """Pretraining preprocessing: uppercase, T->U, non-ACGU -> N."""
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in "ACGU" else "N" for c in seq)


def _payload_lines(path):
    return [l for l in open(path) if not l.lstrip().startswith("#")]


def dot_bracket_to_mat(db):
    n = len(db)
    m = np.zeros((n, n), dtype=np.float32)
    stacks = {c: [] for c in "([{<"}
    for i, c in enumerate(db):
        if c in stacks:
            stacks[c].append(i)
        elif c in _CLOSE_TO_OPEN:
            m[i, stacks[_CLOSE_TO_OPEN[c]].pop()] = 1.0
        elif c != ".":
            raise ValueError(f"unexpected dot-bracket symbol {c!r} at {i} in {path_hint(db)}")
    return np.minimum(m + m.T, 1.0)


def path_hint(db):
    return f"<dbn len {len(db)}>"


def parse_st(path):
    """bpRNA .st: after dropping '#' comments, line 0 = sequence, line 1 = dot-bracket."""
    lines = _payload_lines(path)
    seq, dbn = lines[0].rstrip(), lines[1].rstrip()
    if len(seq) != len(dbn):
        raise ValueError(f"{path}: seq len {len(seq)} != dbn len {len(dbn)}")
    return normalize_seq(seq), dot_bracket_to_mat(dbn)


def parse_bpseq(path):
    """Rivas .bpseq: '<idx> <nt> <pair_idx>' per line, 1-indexed, pair_idx 0 = unpaired."""
    lines = [l for l in _payload_lines(path) if l.strip()]
    n = len(lines)
    seq, m = [], np.zeros((n, n), dtype=np.float32)
    for line in lines:
        i, nt, j = line.split()[:3]
        seq.append(nt)
        i, j = int(i), int(j)
        if j:
            m[i - 1, j - 1] = 1.0
    return normalize_seq("".join(seq)), np.minimum(m + m.T, 1.0)


_SS_PARSERS = {".ct": None, ".st": parse_st, ".bpseq": parse_bpseq}


def load_ss_dir(data_dir, max_len, ct_parser):
    """Load every .ct/.st/.bpseq under data_dir. Returns (items, n_skipped_too_long)."""
    _SS_PARSERS[".ct"] = ct_parser
    items, skipped, bad = [], 0, 0
    for path in sorted(Path(data_dir).rglob("*")):
        fn = _SS_PARSERS.get(path.suffix)
        if fn is None and path.suffix != ".ct":
            continue
        try:
            seq, pair = fn(path)
        except Exception:
            bad += 1
            continue
        if 0 < len(seq) <= max_len:
            items.append({"id": path.stem, "seq": seq, "pair": pair})
        else:
            skipped += 1
    if bad:
        print(f"  [warn] {bad} unparsable files in {data_dir}")
    return items, skipped
