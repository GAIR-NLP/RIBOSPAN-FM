# Copyright 2026 Beijing Academy of Artificial Intelligence (BAAI)
# Apache-2.0

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import torch
from torch import Tensor, nn
from .modules import InteractionLMHead, LMHead, STAR_LayerNorm, TransformerLayer


class _RiboGRAMBackbone(nn.Module):
    """Shared encoder implementation without an additional state-dict prefix."""

    def __init__(
        self,
        alphabet_rna: Mapping[str, int],
        num_layers: int,
        embed_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        required = {
            "rna_alphabet_size",
            "rna_padding_idx",
            "rna_embedding_size",
        }
        missing = required.difference(alphabet_rna)
        if missing:
            raise ValueError(f"alphabet_rna is missing keys: {sorted(missing)}")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if embed_dim < 1 or attention_heads < 1:
            raise ValueError("embed_dim and attention_heads must be positive")
        if embed_dim % attention_heads:
            raise ValueError("embed_dim must be divisible by attention_heads")

        self.rna_alphabet_size = int(alphabet_rna["rna_alphabet_size"])
        self.rna_padding_idx = int(alphabet_rna["rna_padding_idx"])
        self.rna_embedding = int(alphabet_rna["rna_embedding_size"])
        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.attention_heads = int(attention_heads)

        if self.rna_embedding != self.embed_dim:
            raise ValueError("rna_embedding_size must equal embed_dim")

    def _init_encoder(self) -> None:
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.attention_heads,
                    layer_number=layer_number,
                )
                for layer_number in range(self.num_layers)
            ]
        )
        self.layer_norm_after = STAR_LayerNorm(self.rna_embedding)

    @staticmethod
    def _validate_tokens(tokens: Tensor) -> None:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch_size, sequence_length]")
        if tokens.shape[1] == 0:
            raise ValueError("tokens must contain at least one position")

    def _encode(
        self,
        tokens: Tensor,
        embedding: nn.Embedding,
        bias: Optional[Tensor],
        repr_layers: Optional[Iterable[int]],
        need_head_weights: bool,
    ) -> Tuple[Tensor, Dict[int, Tensor], Optional[Tensor]]:
        self._validate_tokens(tokens)
        requested_layers = set(repr_layers or ())
        invalid_layers = requested_layers.difference(range(self.num_layers + 1))
        if invalid_layers:
            raise ValueError(
                f"repr_layers contains invalid layer indices: {sorted(invalid_layers)}"
            )

        padding_mask: Optional[Tensor] = tokens.eq(self.rna_padding_idx)
        features = embedding(tokens)
        features = features * (
            1 - padding_mask.unsqueeze(-1).type_as(features)
        )

        hidden_representations: Dict[int, Tensor] = {}
        if 0 in requested_layers:
            hidden_representations[0] = features

        features = features.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None

        collected_attention = []
        for layer_index, layer in enumerate(self.layers, start=1):
            features, attention = layer(
                features,
                bias=bias,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if layer_index in requested_layers:
                hidden_representations[layer_index] = features.transpose(0, 1)
            if need_head_weights:
                if attention is None:
                    raise RuntimeError("Attention weights were requested but not returned")
                collected_attention.append(attention.transpose(1, 0))

        features = self.layer_norm_after(features).transpose(0, 1)
        if self.num_layers in requested_layers:
            hidden_representations[self.num_layers] = features

        attentions: Optional[Tensor] = None
        if need_head_weights:
            attentions = torch.stack(collected_attention, dim=1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]

        return features, hidden_representations, attentions


class RiboGRAM(_RiboGRAMBackbone):
    """Base nine-token RiboGRAM masked RNA language model."""

    def __init__(
        self,
        alphabet_rna: Optional[Mapping[str, int]] = None,
        num_layers: int = 12,
        embed_dim: int = 1280,
        attention_heads: int = 20,
    ) -> None:
        if alphabet_rna is None:
            alphabet_rna = {
                "rna_alphabet_size": 9,
                "rna_padding_idx": 1,
                "rna_embedding_size": embed_dim,
            }
        super().__init__(
            alphabet_rna=alphabet_rna,
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
        )

        self.rna_tokens = nn.Embedding(
            self.rna_alphabet_size,
            self.rna_embedding,
            padding_idx=self.rna_padding_idx,
        )
        self._init_encoder()
        self.rna_lm_head = LMHead(
            embed_dim=self.rna_embedding,
            output_dim=self.rna_alphabet_size,
        )

    def forward(
        self,
        rna_seq: Tensor,
        bias: Optional[Tensor] = None,
        repr_layers: Optional[Iterable[int]] = None,
        need_head_weights: bool = False,
    ) -> Dict[str, Any]:
        representation, hidden, attentions = self._encode(
            tokens=rna_seq,
            embedding=self.rna_tokens,
            bias=bias,
            repr_layers=repr_layers,
            need_head_weights=need_head_weights,
        )
        result: Dict[str, Any] = {
            "logits": self.rna_lm_head(representation),
            "representation": representation,
            "hidden_representations": hidden,
        }
        if attentions is not None:
            result["attentions"] = attentions
        return result


class RiboGRAMInteraction(_RiboGRAMBackbone):
    """Shared model for RiboGRAM-RPI/RPIplus, RRI and RCI checkpoints."""

    def __init__(
        self,
        alphabet_rna: Optional[Mapping[str, int]] = None,
        num_layers: int = 12,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        expanded_vocab_size: int = 38,
    ) -> None:
        if alphabet_rna is None:
            alphabet_rna = {
                "rna_alphabet_size": 9,
                "rna_padding_idx": 1,
                "rna_embedding_size": embed_dim,
            }
        super().__init__(
            alphabet_rna=alphabet_rna,
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
        )
        if expanded_vocab_size < self.rna_alphabet_size:
            raise ValueError(
                "expanded_vocab_size cannot be smaller than the RNA vocabulary"
            )
        self.expanded_vocab_size = int(expanded_vocab_size)

        # Both embeddings are retained because both are present in the released
        # post-training checkpoints. Forward passes use ``rna_tokens_expend``.
        self.rna_tokens = nn.Embedding(
            self.rna_alphabet_size,
            self.rna_embedding,
            padding_idx=self.rna_padding_idx,
        )
        self.rna_tokens_expend = nn.Embedding(
            self.expanded_vocab_size,
            self.rna_embedding,
            padding_idx=self.rna_padding_idx,
        )
        self._init_encoder()
        self.rna_lm_head = InteractionLMHead(
            embed_dim=self.rna_embedding,
            rna_output_dim=self.rna_alphabet_size,
            expanded_output_dim=self.expanded_vocab_size,
        )
        self.pair_head = nn.Linear(self.rna_embedding, 1)

    def forward(
        self,
        tokens: Tensor,
        bias: Optional[Tensor] = None,
        repr_layers: Optional[Iterable[int]] = None,
        need_head_weights: bool = False,
    ) -> Dict[str, Any]:
        representation, hidden, attentions = self._encode(
            tokens=tokens,
            embedding=self.rna_tokens_expend,
            bias=bias,
            repr_layers=repr_layers,
            need_head_weights=need_head_weights,
        )
        rna_logits, all_logits = self.rna_lm_head(representation)
        pair_logits = self.pair_head(representation[:, 0, :])

        result: Dict[str, Any] = {
            "logits": rna_logits,
            "all_logits": all_logits,
            "representation": representation,
            "hidden_representations": hidden,
            "pair_logits": pair_logits,
        }
        if attentions is not None:
            result["attentions"] = attentions
        return result