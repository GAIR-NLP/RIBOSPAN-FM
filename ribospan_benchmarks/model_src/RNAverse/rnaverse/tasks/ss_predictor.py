from .ss_resnet import renet_b16
import torch.nn as nn
import torch



class SSCNNPredictor(nn.Module):
    def __init__(self, args, extractor, config, tokenizer, is_freeze = False):
        super(SSCNNPredictor, self).__init__()
        self.extractor = extractor
        self.config = config
        self.tokenizer = tokenizer
        self.cnn = renet_b16(myChannels=config.hidden_size, bbn=16)
        self.args = args

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        # RNAverse model
        input_ids = data_dict['input_ids']
        input_struct_tokens = data_dict.get('input_struct_tokens', None)
        input_probe_method_tokens = data_dict.get('input_probe_method_tokens', None)
        input_func_tokens = data_dict.get('input_func_tokens', None)
        attention_mask = data_dict['attention_mask']
        
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(
                    input_ids=input_ids,
                    input_struct_tokens=input_struct_tokens,
                    input_probe_method_tokens=input_probe_method_tokens,
                    input_func_tokens=input_func_tokens,
                    attention_mask=attention_mask
                )
        else:
            output = self.extractor(
                input_ids=input_ids,
                input_struct_tokens=input_struct_tokens,
                input_probe_method_tokens=input_probe_method_tokens,
                input_func_tokens=input_func_tokens,
                attention_mask=attention_mask
            )
        
        # RNAverse returns last_hidden_state
        hidden_states = output.last_hidden_state
  
        ## L*ch-> LxL*ch
        weight_mask = data_dict['weight_mask'] #[bz,ori_max_len] for CLS only
        post_token_length = data_dict['post_token_length']

        assert weight_mask.shape==post_token_length.shape
        mapping_hidden_states = hidden_states
            
        hidden_states = mapping_hidden_states * weight_mask.unsqueeze(2)
        # RNAverse only has CLS token, no SEP token, so remove only CLS (index 0)
            
        matrix = torch.einsum('ijk,ilk->ijlk', hidden_states, hidden_states)

        matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d

        x = self.cnn(matrix)

        x = x.squeeze(-1)

        return x
