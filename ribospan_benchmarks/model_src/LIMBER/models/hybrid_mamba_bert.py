"""Transformer + Mamba hybrid encoder for masked language modeling.

Drop-in replacement for the vanilla BERT backbone used by the kmer1 RNA MLM
baseline. ONLY the encoder backbone changes: the embeddings, MLM head, MLM
objective, tokenizer, and data are identical to the BERT baseline (the
Transformer layers and embeddings/head reuse HuggingFace BERT modules, so they
are bit-for-bit the same blocks).

Encoder = a mix of BERT Transformer layers ('T') and Mamba layers ('M') in a
configurable pattern, default "TTMTTM" for 6 layers (every 3rd layer is Mamba:
2 Transformer + 1 Mamba per group).

Design notes (recorded per task requirements):
  - Transformer block: HuggingFace `BertLayer` -> POST-LayerNorm, gelu,
    same hidden/heads/intermediate/dropout as the BERT baseline. Uses the
    attention_mask.
  - Mamba block: PRE-LayerNorm residual: `x + dropout(Mamba(LayerNorm(x)))`,
    using `mamba_ssm.modules.mamba_simple.Mamba`. Mamba is CAUSAL and does NOT
    consume the attention_mask. Because padding is on the right and Mamba scans
    left->right, trailing pad tokens cannot affect the (earlier) real-token
    representations, and pad positions carry labels=-100 so never enter the
    loss. The bidirectional Transformer layers supply right-context mixing.
  - MLM head: HuggingFace `BertOnlyMLMHead` (dense -> gelu -> LayerNorm ->
    decoder). Decoder is UNTIED by default (tie_word_embeddings configurable).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertOnlyMLMHead,
)

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    _MAMBA_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    Mamba = None
    _MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = _e


class MambaBlock(nn.Module):
    """Pre-LN residual Mamba block: x + dropout(Mamba(LayerNorm(x)))."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1,
                 layer_norm_eps=1e-12):
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Install with:\n"
                "  pip install causal-conv1d mamba-ssm\n"
                f"(original import error: {_MAMBA_IMPORT_ERROR})")
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                           expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        # attention_mask intentionally unused (Mamba is causal; see module docstring)
        return hidden_states + self.dropout(self.mamba(self.norm(hidden_states)))


class HybridMambaConfig(BertConfig):
    """BertConfig + hybrid-backbone fields. `num_hidden_layers` must equal the
    length of `layer_pattern`."""

    model_type = "hybrid_mamba_bert"

    def __init__(self, layer_pattern="TTMTTM", mamba_d_state=16, mamba_d_conv=4,
                 mamba_expand=2, tie_word_embeddings=False,
                 attn_implementation="sdpa", **kwargs):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.layer_pattern = layer_pattern.upper()
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        # transformer attention kernel: "sdpa" (FlashAttention, low-memory) or "eager".
        # identical math to eager; only changes speed/memory.
        self._attn_implementation = attn_implementation
        # keep num_hidden_layers consistent with the pattern
        self.num_hidden_layers = len(self.layer_pattern)


class HybridMambaForMaskedLM(nn.Module):
    """MLM model with a Transformer+Mamba hybrid encoder, BERT embeddings/head."""

    def __init__(self, config: HybridMambaConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)

        layers = []
        for ch in config.layer_pattern:
            if ch == "T":
                layers.append(BertLayer(config))
            elif ch == "M":
                layers.append(MambaBlock(
                    d_model=config.hidden_size,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    dropout=config.hidden_dropout_prob,
                    layer_norm_eps=config.layer_norm_eps,
                ))
            else:
                raise ValueError(f"layer_pattern must be T/M, got {ch!r}")
        self.layers = nn.ModuleList(layers)
        self.is_mamba = [ch == "M" for ch in config.layer_pattern]

        self.cls = BertOnlyMLMHead(config)
        if config.tie_word_embeddings:
            self.cls.predictions.decoder.weight = \
                self.embeddings.word_embeddings.weight

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    @staticmethod
    def _bert_layer_call(layer, hidden, ext_mask):
        return layer(hidden, attention_mask=ext_mask)[0]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                token_type_ids=None, **kwargs):
        hidden = self.embeddings(input_ids=input_ids,
                                 token_type_ids=token_type_ids)

        ext_mask = None
        if attention_mask is not None:
            ext_mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype))
            ext_mask = ext_mask * torch.finfo(hidden.dtype).min

        use_ckpt = self.gradient_checkpointing and self.training
        for layer, is_m in zip(self.layers, self.is_mamba):
            if is_m:
                if use_ckpt:
                    hidden = checkpoint(layer, hidden, use_reentrant=False)
                else:
                    hidden = layer(hidden)                   # Mamba (no mask)
            else:
                if use_ckpt:
                    hidden = checkpoint(self._bert_layer_call, layer, hidden,
                                        ext_mask, use_reentrant=False)
                else:
                    hidden = layer(hidden, attention_mask=ext_mask)[0]  # BertLayer

        logits = self.cls(hidden)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)

    # convenience: parameter breakdown for the param-count script
    def param_groups(self):
        emb = sum(p.numel() for p in self.embeddings.parameters())
        enc = sum(p.numel() for p in self.layers.parameters())
        head = sum(p.numel() for p in self.cls.parameters())
        if self.config.tie_word_embeddings:
            # decoder weight shared with embeddings -> don't double count
            head -= self.embeddings.word_embeddings.weight.numel()
        return {"embedding": emb, "encoder": enc, "mlm_head": head,
                "total": sum(p.numel() for p in self.parameters())}
