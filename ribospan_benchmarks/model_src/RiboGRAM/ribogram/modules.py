# Copyright 2026 Beijing Academy of Artificial Intelligence (BAAI)
# Apache-2.0

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multihead_attention import MultiheadAttention


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SwiGLU(nn.Module):
    def __init__(
        self,
        size_in: int,
        size_out: int,
        beta_is_learnable: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(
            torch.ones(1), requires_grad=beta_is_learnable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_output = self.linear(x)
        swish_output = linear_output * torch.sigmoid(self.beta * linear_output)
        return swish_output * self.linear_gate(x)


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class STAR_LayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not x.is_cuda:
                return super().forward(x)
            with torch.cuda.device(x.device):
                return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as STAR_LayerNorm


class TransformerLayer(nn.Module):
    """RiboGRAM layer combining self-attention and multi-scale convolutions."""

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        layer_number: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.layer_number = layer_number

        self.self_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            layer_number,
        )
        self.attn_layer_norm = STAR_LayerNorm(embed_dim)

        self.kernel_sizes = [1, 3, 5, 7, 9]
        self.K = len(self.kernel_sizes)
        self.dynamic_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.gate_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=self.K,
            kernel_size=1,
            bias=True,
        )
        self.conv_layer_norm = STAR_LayerNorm(embed_dim)

        for convolution in self.dynamic_convs:
            nn.init.xavier_uniform_(convolution.weight, gain=1.0)
        nn.init.xavier_uniform_(self.gate_conv.weight, gain=1.0)
        if self.gate_conv.bias is not None:
            nn.init.constant_(self.gate_conv.bias, 0.0)

        self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        self.ffn_layer_norm = STAR_LayerNorm(embed_dim)
        self.swi_glu = SwiGLU(
            size_in=embed_dim,
            size_out=ffn_embed_dim,
            beta_is_learnable=True,
            bias=True,
        )
        self.ffn_out_proj = nn.Linear(ffn_embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x_rna: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Long-range branch.
        normalized_attention_input = self.attn_layer_norm(x_rna)
        attention_output, attention_weights = self.self_attn(
            query=normalized_attention_input,
            key=normalized_attention_input,
            value=normalized_attention_input,
            bias=bias,
            key_padding_mask=self_attn_padding_mask,
            need_head_weights=need_head_weights,
        )
        long_features = x_rna + attention_output

        # Short-range, multi-scale convolution branch.
        normalized_convolution_input = self.conv_layer_norm(x_rna)
        convolution_input = normalized_convolution_input.permute(1, 2, 0).contiguous()

        padding_mask = self_attn_padding_mask
        if padding_mask is not None:
            convolution_input = convolution_input.masked_fill(
                padding_mask[:, None, :], 0.0
            )

        gate_logits = self.gate_conv(convolution_input)
        if padding_mask is not None:
            expanded_padding_mask = padding_mask[:, None, :].expand(
                -1, self.K, -1
            )
            gate_logits = gate_logits.masked_fill(
                expanded_padding_mask, float("-inf")
            )

        gate_weights = F.softmax(gate_logits, dim=1)
        if padding_mask is not None:
            gate_weights = gate_weights.masked_fill(expanded_padding_mask, 0.0)

        convolution_outputs = [
            convolution(convolution_input) for convolution in self.dynamic_convs
        ]
        convolution_stack = torch.stack(convolution_outputs, dim=1)
        short_features = (
            convolution_stack * gate_weights.unsqueeze(2)
        ).sum(dim=1)
        short_features = short_features.permute(2, 0, 1)

        if padding_mask is not None:
            short_features = short_features.masked_fill(
                padding_mask.transpose(0, 1).unsqueeze(-1), 0.0
            )
        short_features = x_rna + short_features

        fused = self.fusion_proj(
            torch.cat((long_features, short_features), dim=-1)
        )
        feed_forward = self.ffn_out_proj(
            self.swi_glu(self.ffn_layer_norm(fused))
        )
        return fused + feed_forward, attention_weights


class LMHead(nn.Module):
    """Masked-language-model head for the nine-token RNA vocabulary."""

    def __init__(self, embed_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = STAR_LayerNorm(embed_dim)
        self.logits = nn.Linear(embed_dim, output_dim)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(gelu(self.dense(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(self.transform(features))


class InteractionLMHead(LMHead):
    """Dual RNA-only and expanded-vocabulary heads used during post-training."""

    def __init__(
        self,
        embed_dim: int,
        rna_output_dim: int,
        expanded_output_dim: int,
    ) -> None:
        super().__init__(embed_dim=embed_dim, output_dim=rna_output_dim)
        # The historical checkpoint name is ``logits1``; keep it unchanged.
        self.logits1 = nn.Linear(embed_dim, expanded_output_dim)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self.transform(features)
        return self.logits(transformed), self.logits1(transformed)
