"""Supervised contact-prediction head (RiNALMo-style), trained on FROZEN LM
embeddings.

H [L,d] -> pairwise representation [L,L,4d] = concat(h_i, h_j, |h_i-h_j|,
h_i*h_j) -> Linear(4d -> pair_dim) -> 2D ResNet -> symmetric logits [L,L].
The diagonal and |i-j|<4 band are masked for loss/eval (valid_pair_mask).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    def __init__(self, c, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class ContactPredictor(nn.Module):
    def __init__(self, hidden_dim, pair_dim=128, hidden2d=128, num_blocks=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(4 * hidden_dim, pair_dim)
        self.input_conv = nn.Conv2d(pair_dim, hidden2d, 1)
        self.blocks = nn.ModuleList([ResidualBlock2D(hidden2d, dropout) for _ in range(num_blocks)])
        self.output_conv = nn.Conv2d(hidden2d, 1, 1)

    def pair_features(self, H):
        L, d = H.shape
        hi = H.unsqueeze(1).expand(L, L, d)
        hj = H.unsqueeze(0).expand(L, L, d)
        pair = torch.cat([hi, hj, (hi - hj).abs(), hi * hj], dim=-1)  # [L,L,4d]
        return self.proj(pair)                                       # [L,L,pair_dim]

    def forward(self, H):
        """H: [L, d] -> symmetric logits [L, L]."""
        x = self.pair_features(H).permute(2, 0, 1).unsqueeze(0)      # [1,pair_dim,L,L]
        x = self.input_conv(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.output_conv(x).squeeze(0).squeeze(0)          # [L,L]
        return (logits + logits.t()) / 2.0                          # enforce symmetry


def valid_pair_mask(L, min_sep=4, device="cpu"):
    """Boolean [L,L] mask of scorable pairs: i<j and j-i>=min_sep (upper tri)."""
    i = torch.arange(L, device=device)
    sep = (i[None, :] - i[:, None])          # j - i
    return sep >= min_sep                     # excludes diagonal and |i-j|<min_sep, upper-tri
