import logging
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

class RNAverseConfig(PretrainedConfig):
    model_type = "rnaverse"

    def __init__(self,
                # sequence modality and other basic settings
                vocab_size=20,
                hidden_size=640,
                num_hidden_layers=30,
                num_attention_heads=20,
                intermediate_size=2560,
                hidden_act='gelu',
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                max_position_embeddings=1025,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
                position_embedding_type='rotary',
                use_cache=True,
                pad_token_id=19,
                mask_token_id=17,
                cls_token_id=18,
                token_dropout=True,
                emb_layer_norm_before=False,
                is_decoder=False,
                add_cross_attention=False,
                # transformers_version='4.34.0',

                # structure modality
                struct_score_bin_num=10,
                struct_score_null_value=-1,
                num_probe_methods=23,

                # function modality
                func_token_pad_len=15,
                num_func_tokens=4487,
                func_pad_idx=4486,
                
                is_seq_only_model=False, # default is RNAverse, if True, will be RNAverse-seq
                **kwargs
                ):
        super().__init__(pad_token_id=pad_token_id, 
                         mask_token_id=mask_token_id, 
                         cls_token_id=cls_token_id, 
                         **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.token_dropout = token_dropout
        self.emb_layer_norm_before = emb_layer_norm_before
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention

        self.struct_score_bin_num = struct_score_bin_num
        self.struct_score_null_value = struct_score_null_value
        self.num_probe_methods = num_probe_methods

        self.func_token_pad_len = func_token_pad_len
        self.num_func_tokens = num_func_tokens
        self.func_pad_idx = func_pad_idx

        self.is_seq_only_model = is_seq_only_model