# Copyright 2026 Beijing Academy of Artificial Intelligence (BAAI)
# Apache-2.0

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rotary_embedding import RotaryEmbedding


class MultiheadAttention(nn.Module):
    """Checkpoint-compatible multi-head self-attention with rotary embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_number: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.qkv_same_dim = True
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

        self.rot_emb = RotaryEmbedding(self.head_dim)
        self.layer_number = layer_number

        # Retained although inactive because it is present in released checkpoints.
        if layer_number < 2:
            self.alpha_head = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_buffer("alpha_head", None)

    def reset_parameters(self) -> None:
        gain = 1 / math.sqrt(2)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Process tensors shaped ``[length, batch, embedding]``.

        ``bias`` is retained for compatibility with the historical API. The
        released checkpoints were trained without adding this external bias.
        """
        del bias

        target_length, batch_size, embed_dim = query.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, got {embed_dim}"
            )

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.contiguous().view(
            target_length, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)
        k = k.contiguous().view(
            -1, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)
        v = v.contiguous().view(
            -1, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)

        source_length = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            expected = (batch_size, source_length)
            if tuple(key_padding_mask.shape) != expected:
                raise ValueError(
                    f"Expected key_padding_mask shape {expected}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )

        q, k = self.rot_emb(q, k)
        attention_logits = torch.bmm(q, k.transpose(1, 2))

        if key_padding_mask is not None:
            attention_logits = attention_logits.view(
                batch_size,
                self.num_heads,
                target_length,
                source_length,
            )
            attention_logits = attention_logits.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool),
                float("-inf"),
            )
            attention_logits = attention_logits.view(
                batch_size * self.num_heads,
                target_length,
                source_length,
            )

        attention_weights_float = F.softmax(
            attention_logits, dim=-1, dtype=torch.float32
        )
        attention_probabilities = F.dropout(
            attention_weights_float.type_as(attention_logits),
            p=self.dropout,
            training=self.training,
        )
        attention = torch.bmm(attention_probabilities, v)
        attention = attention.transpose(0, 1).contiguous().view(
            target_length, batch_size, embed_dim
        )
        attention = self.out_proj(attention)

        head_weights: Optional[Tensor] = None
        if need_head_weights:
            head_weights = attention_weights_float.view(
                batch_size,
                self.num_heads,
                target_length,
                source_length,
            ).type_as(attention).transpose(1, 0)

        return attention, head_weights
