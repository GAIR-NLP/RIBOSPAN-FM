# Copyright 2026 Beijing Academy of Artificial Intelligence (BAAI)
# Apache-2.0

from typing import Optional, Tuple
import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    length = x.shape[-2]
    return (x * cos[:, :length]) + (rotate_half(x) * sin[:, :length])


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding used by the released checkpoints."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("Rotary embedding dimension must be even")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached: Optional[int] = None
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None

    def _update_cos_sin_tables(
        self, x: Tensor, seq_dimension: int = 1
    ) -> Tuple[Tensor, Tensor]:
        sequence_length = x.shape[seq_dimension]
        cache_is_stale = (
            sequence_length != self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != x.device
        )
        if cache_is_stale:
            self._seq_len_cached = sequence_length
            positions = torch.arange(sequence_length, device=x.device).type_as(
                self.inv_freq
            )
            frequencies = torch.einsum("i,j->ij", positions, self.inv_freq)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            self._cos_cached = embedding.cos()[None, :, :]
            self._sin_cached = embedding.sin()[None, :, :]

        assert self._cos_cached is not None
        assert self._sin_cached is not None
        return self._cos_cached, self._sin_cached

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        cos, sin = self._update_cos_sin_tables(key, seq_dimension=-2)
        return apply_rotary_pos_emb(query, cos, sin), apply_rotary_pos_emb(
            key, cos, sin
        )
