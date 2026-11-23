# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Residual convolutional pairwise head from the RiNALMo secondary-structure pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn


def _outer_concat(states: torch.Tensor) -> torch.Tensor:
    length = states.shape[1]
    left = states.unsqueeze(-2).expand(-1, -1, length, -1)
    right = states.unsqueeze(-3).expand(-1, length, -1, -1)
    return torch.cat((left, right), dim=-1)


class _ResNet2DBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, bias=False, padding="same"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states + self.net(states)


class SecStructPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, num_blocks: int = 2, conv_dim: int = 64, kernel_size: int = 3):
        super().__init__()
        self.linear_in = nn.Linear(embed_dim * 2, conv_dim)
        self.resnet = nn.ModuleList(
            [_ResNet2DBlock(conv_dim, kernel_size) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size, padding="same")

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.linear_in(_outer_concat(states))
        hidden = hidden.permute(0, 3, 1, 2)
        for block in self.resnet:
            hidden = block(hidden)
        logits = self.conv_out(hidden).squeeze(-3)
        upper = torch.triu(logits, diagonal=1)
        return upper + upper.transpose(-1, -2)
