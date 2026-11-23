# Copyright (c) 2026 RNAverse. All rights reserved.
# Licensed under the MIT License.
#
# This file is derived from the BERT and ESM model implementation in Hugging Face Transformers.

# Original Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

from .configuration_rnaverse import RNAverseConfig
from .modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    RNAverseMaskedLMOutput,
    SequenceClassifierOutput
)
from transformers import PreTrainedModel

import torch.nn.functional as F

# Check if SDPA is available (PyTorch >= 2.0)
try:
    from torch.nn.functional import scaled_dot_product_attention
    sdpa_available = True
except ImportError:
    sdpa_available = False

# When adding FA2, refer to the following URL:
# https://github.com/huggingface/transformers/pull/26585/files#diff-a48ba7f6444ca4954a58f1ac3e66c7941a2bbc4615649d56b182aeac8cc36d9c
# test import flash attention related modules
# if not successful, just ignore and continue
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    flash_attention_installed = True
except ImportError:
    flash_attention_installed = False

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def average_product_correct(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

@torch.fx.wrap
def check_padding_in_attention_mask(attention_mask):
    if 0 in attention_mask:
        return attention_mask
    return None

class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            
        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )

class RNAverseEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

        self.is_seq_only_model = config.is_seq_only_model
        if self.is_seq_only_model == False:
            self.struct_embeddings = nn.Embedding(config.struct_score_bin_num + 1, config.hidden_size) # +1 for null/masked value
            self.probe_method_embeddings = nn.Embedding(config.num_probe_methods, config.hidden_size)
            self.func_embeddings = nn.EmbeddingBag(config.num_func_tokens, config.hidden_size, mode='sum', padding_idx=config.func_pad_idx)

    def forward(
        self, input_ids=None, input_struct_tokens=None, input_probe_method_tokens=None, input_func_tokens=None,
        attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        # although for the 2nd stage, multimodal alignment, the mask ratio is betalinear30
        # we still use the hardcoded code for esm here: Finetuning help the model to fit into the scaled embedding.
        if self.token_dropout and input_ids is not None:
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
                embeddings.dtype
            )

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        
        if self.is_seq_only_model == False:
            struct_embeddings = self.struct_embeddings(input_struct_tokens)
            probe_method_embeddings = self.probe_method_embeddings(input_probe_method_tokens)
            struct_embeddings = struct_embeddings + probe_method_embeddings

            embeddings[:, 1:, :] = embeddings[:, 1:, :] + struct_embeddings

            func_embed = self.func_embeddings(input_func_tokens) # input_func_tokens shape is bs * func_token_pad_len
            # func_embed shape is bs * hidden_size
            embeddings = embeddings + func_embed.unsqueeze(1)
        
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        # Matt: I think this line was copied incorrectly from BERT, disabling it for now.
        # embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

class RNAverseSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    

    def _query_key_value_attention_mask(self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer * self.attention_head_size**-0.5
        return query_layer, key_layer, value_layer, attention_mask
        # Matt: Our BERT model (which this code was derived from) scales attention logits down by sqrt(head_dim).
        # ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent,
        # but not when rotary embeddings get involved. Therefore, we scale the query here to match the original
        # ESM code and fix rotary embeddings.
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:    
        query_layer, key_layer, value_layer, attention_mask = self._query_key_value_attention_mask(hidden_states,attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value)
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in EsmModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class RNAverseSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class RNAverseAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RNAverseSelfAttention(config, position_embedding_type)
        self.output = RNAverseSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            padding_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RNAverseIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class RNAverseOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class RNAverseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = RNAverseAttention(config=config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RNAverseAttention(config)
        self.intermediate = RNAverseIntermediate(config)
        self.output = RNAverseOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            padding_mask=padding_mask,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RNAverseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RNAverseLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # Attention mask is inverted in BertModel forward,
        padding_mask = check_padding_in_attention_mask((attention_mask==0).float())
        if padding_mask is not None:
            padding_mask = padding_mask.squeeze(1).squeeze(1)

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    padding_mask
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RNAversePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RNAversePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RNAverseConfig
    base_model_prefix = "rnaverse"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RNAverseLayer", "RNAverseEmbeddings"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RNAverseEncoder):
            module.gradient_checkpointing = value


class RNAverseModel(RNAversePreTrainedModel):
    def __init__(self, config, add_pooling_layer=True): 

        super().__init__(config)
        self.config = config

        self.embeddings = RNAverseEmbeddings(config)
        self.encoder = RNAverseEncoder(config)
        self.pooler = RNAversePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.Tensor] = None,
        input_probe_method_tokens: Optional[torch.Tensor] = None,
        input_func_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Forward pass for the RNAverse model, integrating sequence, structure, and function.

        Args:
            input_ids (torch.LongTensor): 
                Indices of input sequence tokens. The first token is always 'CLS' (index 18),
                representing the global contextual embedding of the RNA molecule.
                RNAverse does not have the SEP token.
            input_struct_tokens (torch.LongTensor, optional): 
                Quantized experimental reactivity bins. These represent the structural 
                score of each nucleotide.
            input_probe_method_tokens (torch.LongTensor, optional): 
                IDs representing the probing assay (e.g., SHAPE-MaP, DMS-seq). 
            input_func_tokens (torch.LongTensor, optional): 
                Indices of functional descriptors (Rfam families, GO terms...). 
                Processed via EmbeddingBag.
            attention_mask (torch.Tensor, optional): 
                Mask to avoid attention on padding (0 for pad, 1 for non-pad).

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions:
                last_hidden_state (torch.Tensor): Final hidden states of the model.
                pooler_output (torch.Tensor): Pooled output from the CLS token.
                past_key_values (List[torch.FloatTensor]): Cached key/values for faster decoding.
                hidden_states (Optional[Tuple[torch.FloatTensor]]): Hidden states from all layers.
                attentions (Optional[Tuple[torch.FloatTensor]]): Attention weights from all layers.
                cross_attentions (Optional[Tuple[torch.FloatTensor]]): Cross-attention weights from all layers.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # allow passing both input_ids and inputs_embeds (inputs_embeds replaces the word embedding,
        # input_ids is still used for token_dropout / position_ids in early fusion)
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class RNAverseForMaskedLM(RNAversePreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config): 
        super().__init__(config)

        self.rnaverse = RNAverseModel(config, add_pooling_layer=False)

        self.lm_head = RNAverseLMHead(config)

        if config.is_seq_only_model is False:
            self.struct_lm_head = RNAverseLMStructHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RNAverseMaskedLMOutput]:
        """
        Forward pass for the RNAverse MaskedLM model, integrating sequence, structure, and function.
        Predict masked sequence and structure tokens.

        Args:
            input_ids (torch.LongTensor): 
                Indices of input sequence tokens. The first token is always 'CLS' (index 18),
                representing the global contextual embedding of the RNA molecule.
                RNAverse does not have the SEP token.
            input_struct_tokens (torch.LongTensor, optional): 
                Quantized experimental reactivity bins. These represent the structural 
                score of each nucleotide.
            input_probe_method_tokens (torch.LongTensor, optional): 
                IDs representing the probing assay (e.g., SHAPE-MaP, DMS-seq). 
            input_func_tokens (torch.LongTensor, optional): 
                Indices of functional descriptors (Rfam families, GO terms...). 
                Processed via EmbeddingBag.
            attention_mask (torch.Tensor, optional): 
                Mask to avoid attention on padding (0 for pad, 1 for non-pad).

        Returns:
            RNAverseMaskedLMOutput:
                loss (torch.FloatTensor, optional): Masked language modeling loss for sequence modality.
                logits (torch.FloatTensor): Prediction scores for masked sequence tokens.
                struct_logits (torch.FloatTensor, optional): Prediction scores for masked structure tokens, 
                    must use sigmoid afterwards to get final structural scores within [0,1].
                hidden_states (Optional[Tuple[torch.FloatTensor]]): Hidden states from all layers.
                attentions (Optional[Tuple[torch.FloatTensor]]): Attention weights from all layers.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rnaverse(
            input_ids,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # bs * (seq_len+1) * hidden_size

        prediction_scores = None
        struct_prediction_scores = None

        prediction_scores = self.lm_head(sequence_output)
        if self.config.is_seq_only_model is False:
            struct_prediction_scores = self.struct_lm_head(sequence_output) # must use sigmoid afterwards to get final structural scores within [0,1]

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return RNAverseMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            struct_logits=struct_prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RNAverseLMStructHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project to 1-d shape value, regression
        x = self.decoder(x) + self.bias
        return x


class RNAverseLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x


class RNAverseForCRISPROffTarget(RNAversePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        self.rnaverse = RNAverseModel(config, add_pooling_layer=True)

        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,

        target_input_ids: Optional[torch.Tensor] = None,
        target_struct_tokens: Optional[torch.LongTensor] = None,
        target_probe_method_tokens: Optional[torch.LongTensor] = None,
        target_func_tokens: Optional[torch.LongTensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        sgrna_out = self.rnaverse(
            input_ids,
            attention_mask=attention_mask,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
        ).pooler_output
        target_out = self.rnaverse(
            target_input_ids,
            attention_mask=target_attention_mask,
            input_struct_tokens=target_struct_tokens,
            input_probe_method_tokens=target_probe_method_tokens,
            input_func_tokens=target_func_tokens,
        ).pooler_output
        final_input = torch.cat([sgrna_out,target_out],dim=-1)
        logits = self.classifier(final_input)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + sgrna_out[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, #sgrna_out.hidden_states,
            attentions=None, #sgrna_out.attentions,
        )
    
class RNAverseForSequenceClassification(RNAversePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.rnaverse = RNAverseModel(config, add_pooling_layer=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #return_dict = None
        outputs = self.rnaverse(
            input_ids=input_ids,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                    # print('single_label_classification')
                else:
                    self.config.problem_type = "multi_label_classification"
         
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
 
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=3):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored
        return score

class RNAverseForNucleotideLevel(RNAversePreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.rnaverse = RNAverseModel(config, add_pooling_layer=False)
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        weight_mask: Optional[bool] = None,
        post_token_length: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.rnaverse(
            input_ids,
            attention_mask=attention_mask,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        final_input= outputs[0]

        if attention_mask.shape != weight_mask.shape or attention_mask.shape != post_token_length.shape:
            raise ValueError("attention_mask(%s), weight_mask(%s) and post_token_length(%s) must have the same shape" % (attention_mask.shape, weight_mask.shape, post_token_length.shape))

        mapping_final_input = final_input
        
        mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)

        logits = self.classifier(mapping_final_input)
   
        loss = None
        if labels is not None:
            logits = logits[:, 1:1+labels.size(1), :]
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MCRMSELoss()
                
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1).long())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RNAverseSeqForStructuralimputation(RNAversePreTrainedModel):

    def __init__(self, config, fusion_mode='late'):
        super().__init__(config)

        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 1
        self.config = config
        self.fusion_mode = fusion_mode
        
        self.rnaverse = RNAverseModel(config, add_pooling_layer=False)
        if config.is_seq_only_model is False:
            raise ValueError("RNAverseSeqForStructuralimputation only works for RNAverse-seq, please set config.is_seq_only_model to True")
        self.struct_embed = nn.Embedding(11, config.hidden_size) # 0-9 for 10 bins, 10 for null/masked value
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.transfer = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        struct: Optional[torch.Tensor] = None,
        weight_mask: Optional[bool] = None,
        post_token_length: Optional[bool] = None,
        l_mask=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.fusion_mode == 'early':
            # early fusion: add the 11-bin struct embedding into the input embeddings via inputs_embeds
            seq_embeds = self.rnaverse.get_input_embeddings()(input_ids)
            struct_embeds = self.struct_embed(struct)
            inputs_embeds = seq_embeds.clone()
            slen = struct_embeds.shape[1]
            inputs_embeds[:, 1:slen+1, :] = inputs_embeds[:, 1:slen+1, :] + struct_embeds
            outputs = self.rnaverse(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs = self.rnaverse(
                input_ids,
                attention_mask=attention_mask,
                input_struct_tokens=input_struct_tokens,
                input_probe_method_tokens=input_probe_method_tokens,
                input_func_tokens=input_func_tokens,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        final_input= outputs[0]


        if attention_mask.shape != weight_mask.shape or attention_mask.shape != post_token_length.shape:
            raise ValueError("attention_mask(%s), weight_mask(%s) and post_token_length(%s) must have the same shape" % (attention_mask.shape, weight_mask.shape, post_token_length.shape))
        mapping_final_input = final_input
        mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)
        mapping_final_input = mapping_final_input[:, 1:, :] # exclude [cls] token

        if self.fusion_mode == 'late':
            struct_embed = self.struct_embed(struct) # struct is the bin token (0-10), shape (bs, seq_len)
            hidden_states = mapping_final_input + struct_embed
        else:
            hidden_states = mapping_final_input
        # same order to that in RNAverseLMHead
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.transfer(hidden_states)
        
        logits = self.classifier(hidden_states)

        label_mask = l_mask
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits[label_mask].squeeze(), labels.squeeze())
      
                else:
                    loss = loss_fct(logits[label_mask], labels)

        if not return_dict:
            output = (logits[label_mask],) + outputs[2:]
            return ((loss,) + output) if loss is not None else output       
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits[label_mask],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class RNAverseForStructuralimputation(RNAversePreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 1
        self.config = config
    
        self.rnaverse = RNAverseModel(config, add_pooling_layer=False)
        if config.is_seq_only_model is True:
            raise ValueError("RNAverseForStructuralimputation only works for RNAverse, not RNAverse-seq, please set config.is_seq_only_model to False")
        self.struct_lm_head = RNAverseLMStructHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        struct: Optional[torch.Tensor] = None,
        weight_mask: Optional[bool] = None,
        post_token_length: Optional[bool] = None,
        l_mask=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.rnaverse(
            input_ids,
            attention_mask=attention_mask,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        final_input= outputs[0]

        if attention_mask.shape != weight_mask.shape or attention_mask.shape != post_token_length.shape:
            raise ValueError("attention_mask(%s), weight_mask(%s) and post_token_length(%s) must have the same shape" % (attention_mask.shape, weight_mask.shape, post_token_length.shape))
        mapping_final_input = final_input
        mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)
        mapping_final_input = mapping_final_input[:, 1:, :]

        logits = torch.sigmoid(self.struct_lm_head(mapping_final_input))

        label_mask = l_mask
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits[label_mask].squeeze(), labels.squeeze())
      
                else:
                    loss = loss_fct(logits[label_mask], labels)

        if not return_dict:
            output = (logits[label_mask],) + outputs[2:]
            return ((loss,) + output) if loss is not None else output       
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits[label_mask],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RNAverseForSequenceClassificationMeanPool(RNAversePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if config.is_seq_only_model is True:
            raise ValueError("RNAverseForSequenceClassificationMeanPool only works for RNAverse, not RNAverse-seq, please set config.is_seq_only_model to False")
        self.rnaverse = RNAverseModel(config, add_pooling_layer=False)
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.transfer = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rnaverse(
            input_ids=input_ids,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        seq_output = outputs[0][:, 1:, :]
        seq_output = self.dense(seq_output)
        seq_output = self.transform_act_fn(seq_output)
        seq_output = self.LayerNorm(seq_output)
        seq_output = self.transfer(seq_output)
        pooled_output = torch.mean(seq_output, dim=1) # mean pool the token representations, ignore the CLS token
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"

                else:
                    self.config.problem_type = "multi_label_classification"
         
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
 
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RNAverseSeqForSequenceClassificationWithStr(RNAversePreTrainedModel):
    def __init__(self, config, fusion_mode='late'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.fusion_mode = fusion_mode

        self.rnaverse = RNAverseModel(config, add_pooling_layer=False)
        self.dense = torch.nn.Linear(config.hidden_size,config.hidden_size)
        self.transform_act_fn = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.transfer = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.struct_embed = nn.Embedding(11, config.hidden_size) # 0-9 for 10 bins, 10 for null/masked value
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        struct: Optional[torch.Tensor] = None,
        input_struct_tokens: Optional[torch.LongTensor] = None,
        input_probe_method_tokens: Optional[torch.LongTensor] = None,
        input_func_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.fusion_mode == 'early':
            # early fusion: add the 11-bin struct embedding into the input embeddings via inputs_embeds
            seq_embeds = self.rnaverse.get_input_embeddings()(input_ids)
            struct_embeds = self.struct_embed(struct)
            inputs_embeds = seq_embeds.clone()
            slen = struct_embeds.shape[1]
            inputs_embeds[:, 1:slen+1, :] = inputs_embeds[:, 1:slen+1, :] + struct_embeds
            outputs = self.rnaverse(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs = self.rnaverse(
                input_ids=input_ids,
                input_struct_tokens=input_struct_tokens,
                input_probe_method_tokens=input_probe_method_tokens,
                input_func_tokens=input_func_tokens,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        

        seq_output = outputs[0][:, 1:, :]
        if self.fusion_mode == 'late':
            # late fusion: add the 11-bin struct embedding before the downstream MLP
            struct_embed = self.struct_embed(struct)
            seq_output = seq_output + struct_embed
        # early fusion already adds the struct in the input embeddings, so no struct added here
        seq_output = self.dense(seq_output)
        seq_output = self.transform_act_fn(seq_output)
        seq_output = self.LayerNorm(seq_output)
        seq_output = self.transfer(seq_output)

        logits = torch.mean(seq_output, dim=1)
        logits = self.classifier(logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"

                else:
                    self.config.problem_type = "multi_label_classification"
         
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
 
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
