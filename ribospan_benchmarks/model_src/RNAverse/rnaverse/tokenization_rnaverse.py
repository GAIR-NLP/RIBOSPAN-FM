import json
import logging
import torch
import os
import numpy as np
from transformers import PreTrainedTokenizer
from .utils import tokenize_struct_score

logger = logging.getLogger(__name__)

class RNAverseTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "seq_vocab.json",
                         "probe_vocab_file": "probe_vocab.json",
                         "func_vocab_file": "func_vocab.json",
                         "rfam_map_file": "rfam_map.json"}
    tokenizer_class = "RNAverseTokenizer"

    def __init__(self,
                 vocab_file,
                 probe_vocab_file,
                 func_vocab_file,
                 rfam_map_file,
                 struct_score_bin_num=10,
                 struct_score_null_value=-1,
                 func_token_pad_len=15,
                 **kwargs):

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.dict_base_id = json.load(f)
        with open(probe_vocab_file, "r", encoding="utf-8") as f:
            self.dict_probe_method_id = json.load(f)
        with open(func_vocab_file, "r", encoding="utf-8") as f:
            self.dict_func_token = json.load(f)
        with open(rfam_map_file, "r", encoding="utf-8") as f:
            self.dict_rfamid_func = json.load(f)
        self.dict_struct_token_id = {str(i):i for i in range(struct_score_bin_num)}
        self.dict_struct_token_id[str(struct_score_null_value)] = struct_score_bin_num

        self.struct_score_bin_num = struct_score_bin_num
        self.struct_score_null_value = struct_score_null_value
        self.func_token_pad_len = func_token_pad_len

        # self.func_pad_idx = func_pad_idx
        self.seq_token_to_id = {v:k for k,v in self.dict_base_id.items() if k != 'T'} # only retain U

        super().__init__(
            **kwargs,
        )
        self.cls_token = 'CLS'
        self.pad_token = 'PAD'
        self.mask_token = 'MASK'
        
        # 更新特殊 token 映射
        self._special_tokens_map = {
            'cls_token': 'CLS',
            'pad_token': 'PAD',
            'mask_token': 'MASK',
        }
    @property
    def vocab_size(self):
        '''
        The vocab size is the size of primary sequence vocab
        '''
        return len(self.dict_base_id)
    
    @property
    def probe_method_vocab_size(self):
        return len(self.dict_probe_method_id)
    
    @property
    def struct_token_vocab_size(self):
        return len(self.dict_struct_token_id)
    
    @property
    def func_token_vocab_size(self):
        return len(self.dict_func_token)

    def _tokenize(self, seq):
        return list(seq.upper())

    def _convert_token_to_id(self, token):
        # This handles the primary sequence conversion
        return self.dict_base_id[token]

    def _convert_id_to_token(self, index):
        # Convert primary seq token ID back to a token string
        return self.seq_token_to_id[index]
    
    def get_vocab(self):
        return self.dict_base_id
    
    def get_probe_method_vocab(self):
        return self.dict_probe_method_id
    
    def get_struct_token_vocab(self):
        return self.dict_struct_token_id

    def get_func_token_vocab(self):
        return self.dict_func_token
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)

        file_map = {"vocab_file": self.dict_base_id,
                    "probe_vocab_file": self.dict_probe_method_id,
                    "func_vocab_file": self.dict_func_token,
                    "rfam_map_file": self.dict_rfamid_func}
        saved_files = []
        for key, vocab_dict in file_map.items():
            file_name = self.vocab_files_names[key]
            vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + file_name)
            with open(vocab_file, "w", encoding="utf-8") as f:
                json.dump(vocab_dict, f, indent=4, ensure_ascii=False)
            saved_files.append(vocab_file)
        return tuple(saved_files)
    
    def _prep_single_input(self, seq, struct_score_list=None, probe_method=None, rfam_id=None, func_list=None, mask_pos_list=None):
        '''
        This function takes raw input for one sample sequence and convert it into token ids and other modalities
        '''
        if struct_score_list is None:
            struct_score_list = [self.struct_score_null_value] * len(seq)
        if probe_method is None:
            probe_method = 'UNK'
        if func_list is None:
            func_list = ['UNK']
        if mask_pos_list is None:
            mask_pos_list = []

        if len(seq) != len(struct_score_list):
            raise ValueError('The length of sequence and structural scores must be the same!')
        
        if len(func_list) > self.func_token_pad_len:
            raise ValueError('The length of func_list is longer than the pad length defined in the func_anno_file!')
        
        seq_for_token = self._tokenize(seq)
        seq_ids = [self._convert_token_to_id(tk) for tk in seq_for_token]

        # Apply mask
        for mask_pos in mask_pos_list:
            seq_ids[mask_pos] = self.dict_base_id['MASK']

        seq_ids = [self.dict_base_id['CLS']] + seq_ids

        probe_method_list = [self.dict_probe_method_id[probe_method]] * len(seq)

        if rfam_id is not None:
            if rfam_id in self.dict_rfamid_func:
                func_list = self.dict_rfamid_func[rfam_id]
            else:
                logger.warning(f"The provided rfam_id {rfam_id} is not found in the rfam_map_file, will use func_list=['UNK']")
                func_list = ['UNK']
                
        func_np = np.ones((self.func_token_pad_len), dtype=int) * self.dict_func_token['PAD']
        for i1, x1 in enumerate(func_list):
            func_np[i1] = self.dict_func_token[x1]

        return {
            'input_ids': seq_ids,
            'input_struct_scores': struct_score_list,
            'input_probe_method_tokens': probe_method_list,
            'input_func_tokens': func_np.tolist(),
            'attention_mask': [1] * len(seq_ids),
        }

    def __call__(self, raw_str_batch, max_length=None, padding=True, truncation=False, return_tensors='pt', 
                 pad_to_max_length=False, **kwargs):

        '''
        Prepare a batch of multimodal RNA input data for RNAverse model.

        Args:
            raw_str_batch (List[Dict]): A list of dictionaries, each containing:
                - 'seq' (str): The raw RNA sequence (e.g., "AUGC...").
                - 'struct_score_list' (List[float], optional): Normalized structural scores (0 to 1). 
                    Missing values should be represented as null values (e.g., -1). Defaults to None.
                - 'probe_method' (str, optional): The experimental probing technology 
                    (e.g., 'icSHAPE', 'DMS-seq'). Defaults to 'UNK'.
                - 'rfam_id' (str, optional): Rfam family ID (e.g., 'RF00001') to derive functional annotations.
                    if specified, it will override the provided 'func_list'. Defaults to None.
                - 'func_list' (List[str], optional): Functional annotations
                    (e.g., Rfam IDs like 'RF00001'). Defaults to ['UNK'].
                - 'mask_pos_list' (List[int], optional): 0-based indices in the sequence 
                    to be replaced with the MASK token.
            max_length (int, optional): Maximum sequence length, typically 1025 (1024 + CLS).
            padding (bool): Whether to pad sequences to the maximum length in the batch.
            truncation (bool): Whether to truncate sequences exceeding `max_length`.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - input_ids: Integer tokens for the RNA sequence [Batch, Seq_Len + 1]. +1 for CLS token
                - input_struct_tokens: tokenized structural scores [Batch, Seq_Len].
                - input_probe_method_tokens: Technology IDs [Batch, Seq_Len].
                - input_func_tokens: Pooled functional annotation IDs [Batch, Func_Pad_Len].
                - attention_mask: Mask to avoid performing attention on padding tokens.
        '''
        
        # first collect the max len in batch, then decide the max len to pad/truncate to
        max_len_in_batch = max([len(x['seq']) for x in raw_str_batch]) + 1 # +1 for CLS token
        if max_length is None:
            max_len = max_len_in_batch
        else:
            if max_len_in_batch > max_length:
                if truncation:
                    max_len = max_length
                else:
                    max_len = max_len_in_batch
                    logger.warning(
                        f"Max sequence length +1 ({max_len_in_batch}) exceeds specified max_length ({max_length}), "
                        f"but truncation=False. Keeping original length {max_len_in_batch}. "
                        f"This may cause memory issue. Consider setting truncation=True if have problems"
                        )
            else:
                if pad_to_max_length:
                    max_len = max_length
                else:
                    max_len = max_len_in_batch

        if padding == False:
            raise ValueError('Please set padding=True, since the model requires a fixed length input')
        
        seq_list = []
        struct_score_list = []
        probe_method_list = []
        atten_mask_list = []
        func_token_list = []

        for x in raw_str_batch:
            single_output = self._prep_single_input(x['seq'], 
                            x.get('struct_score_list', None),
                            x.get('probe_method', None),
                            x.get('rfam_id', None),
                            x.get('func_list', None), 
                            x.get('mask_pos_list', None))
            # 'input_ids' max_len 
            # 'input_struct_scores' max_len - 1
            # 'input_probe_method_tokens' max_len - 1
            # 'input_func_tokens' no need to adjust
            # 'attention_mask' max_len

            single_output['input_ids'] = single_output['input_ids'][:max_len]
            single_output['input_struct_scores'] = single_output['input_struct_scores'][:max_len-1]
            single_output['input_probe_method_tokens'] = single_output['input_probe_method_tokens'][:max_len-1]
            single_output['attention_mask'] = single_output['attention_mask'][:max_len]

            seq_ind = single_output['input_ids'] + [self.dict_base_id['PAD']] * (max_len - len(single_output['input_ids']))
            seq_list.append(torch.tensor(seq_ind))
            
            struct_ind = single_output['input_struct_scores'] + [self.struct_score_null_value] * (max_len - 1 - len(single_output['input_struct_scores']))
            struct_score_list.append(torch.tensor(struct_ind))
            
            probe_method_ind = [single_output['input_probe_method_tokens'][0]] * (max_len - 1)
            probe_method_list.append(torch.tensor(probe_method_ind))

            func_token_list.append(torch.tensor(single_output['input_func_tokens']))

            atten_ind = single_output['attention_mask'] + [0] * (max_len - len(single_output['attention_mask']))
            atten_mask_list.append(torch.tensor(atten_ind))

        # struct_all = torch.cat([x1.unsqueeze(0) for x1 in struct_score_list], 0)
        struct_all = torch.stack(struct_score_list, dim=0)
        struct_token = tokenize_struct_score(struct_all, bin_num=self.struct_score_bin_num, null_value=self.struct_score_null_value)


        batch_data = {}
        batch_data['input_ids'] = torch.stack(seq_list, dim=0).long()
        batch_data['input_struct_tokens'] = struct_token
        batch_data['input_probe_method_tokens'] = torch.stack(probe_method_list, dim=0).long()
        batch_data['input_func_tokens'] = torch.stack(func_token_list, dim=0).long()
        batch_data['attention_mask'] = torch.stack(atten_mask_list, dim=0).float()
        
        return batch_data
        