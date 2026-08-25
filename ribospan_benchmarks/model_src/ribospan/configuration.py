# coding=utf-8
# Copyright (c) 2026 RIBOSPAN Team Authors.
#
# Copyright 2021- NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RIBOSPAN model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class RiboSpanConfig(PretrainedConfig):
    """Configuration for RIBOSPAN."""

    model_type = "ribospan"

    def __init__(
        self,
        vocab_size=16,
        hidden_size=2048,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=5440,
        hidden_act="swiglu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=10240,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=0,
        add_linear_bias=True,
        position_embedding_type="rope",
        normalization_type="LayerNorm",
        use_cache=True,
        rotary_percent=1.0,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.add_linear_bias = add_linear_bias
        self.normalization_type = normalization_type
        self.rotary_percent = rotary_percent
        self.rope_scaling = rope_scaling
