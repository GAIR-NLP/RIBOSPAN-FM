"""
Inference script to extract embeddings from RNAverse models.
input_txt example is ./examples/input_demo.txt
Usage:
    python infer_embed.py \
        --model_name_or_path /path/to/model \
        --input_txt /path/to/input.txt \
        --output_dir /path/to/output \
        --output_embed_format ave \
        --batch_size 64

Input file format (tab-separated):
    name + probing_tech + func_annotation + sequence + structure_scores...
    Example: seq1\tSHAPE-MaP\ttRNA,RF00005\tACGUACGU\t0.1\t0.2\tNULL\t0.3...
"""
import argparse
import os
import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from rnaverse import RNAverseForMaskedLM, RNAverseTokenizer
from rnaverse.utils import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract embeddings from RNAverse models')
    
    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to pretrained model or model identifier from huggingface.co/models')
    
    # Data arguments
    parser.add_argument('--input_txt', type=str, required=True,
                        help='Input txt file (tab-separated: name, probing_tech, func_anno, seq, structure_scores...)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for embeddings')
    parser.add_argument('--output_embed_format', type=str, default='ave',
                        choices=['ave', 'all'],
                        help='Output embedding format: "ave" for average pooling, "all" for all positions')
    parser.add_argument('--max_seq_num_in_output_file', type=int, default=200000,
                        help='Maximum number of sequences per output file')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--random_seed', type=int, default=37,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    return args


class IterDataset(IterableDataset):
    """Iterable dataset for loading sequences from tab-separated file."""
    
    def __init__(self, input_txt):
        self.input_txt = input_txt
    
    def __iter__(self):
        with open(self.input_txt, 'r') as f:
            for line in f:
                tmp_list = line.rstrip('\n').split('\t')
                # Format: name + tab + probing_tech + tab + func_anno + tab + sequence + tab + structure_scores
                name = tmp_list[0]
                str_prob_tech = tmp_list[1] if len(tmp_list) > 1 else 'UNK'
                func_anno = tmp_list[2] if len(tmp_list) > 2 else ''
                func_list = [x.strip() for x in func_anno.split(',') if x.strip()] if func_anno else ['UNK']
                seq = tmp_list[3] if len(tmp_list) > 3 else tmp_list[0]
                
                # Parse structure scores
                struct_score_list = None
                if len(tmp_list) > 4:
                    shape_list_raw = tmp_list[4:]
                    struct_score_list = []
                    for x in shape_list_raw:
                        if x == 'NULL' or x == '':
                            struct_score_list.append(-1)
                        else:
                            struct_score_list.append(float(x))
                
                yield {
                    'seq': seq,
                    'struct_score_list': struct_score_list,
                    'probe_method': str_prob_tech,
                    'func_list': func_list
                }


def collate_fn_batch(batch, tokenizer):
    """Collate function for DataLoader using RNAverseTokenizer."""
    # Prepare input dict list for tokenizer
    input_dict_list = [
        {
            'seq': item['seq'],
            'struct_score_list': item.get('struct_score_list', None),
            'probe_method': item.get('probe_method', None),
            'func_list': item.get('func_list', None)
        }
        for item in batch
    ]
    
    # Tokenize using RNAverseTokenizer
    # Returns: input_ids, input_struct_tokens, input_probe_method_tokens, input_func_tokens, attention_mask
    inputs = tokenizer(input_dict_list, padding=True, return_tensors='pt')
    return inputs


def main(args):
    seed_everything(args.random_seed)
    
    tick1 = time.time()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Count sequences
    line_num = 0
    with open(args.input_txt, 'r') as f:
        for line in f:
            line_num += 1
    print(f'Sequence number in input txt: {line_num}')
    tqdm_num = (line_num + args.batch_size - 1) // args.batch_size
    
    # Load tokenizer
    print(f"Loading tokenizer from: {args.model_name_or_path}")
    tokenizer = RNAverseTokenizer.from_pretrained(args.model_name_or_path)
    
    # Load model
    print(f"Loading model from: {args.model_name_or_path}")
    
    model = RNAverseForMaskedLM.from_pretrained(
        args.model_name_or_path,
    )
    model.eval()
    model.to(args.device)
    
    # Prepare dataset
    dataset = IterDataset(args.input_txt)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn_batch(batch, tokenizer)
    )
    
    # Extract embeddings
    res_list = []
    count_file_num = 0
    
    print("Extracting embeddings...")
    for i1, x1 in tqdm(enumerate(dataloader), total=tqdm_num):
        with torch.no_grad():
            inputs = {k: v.to(args.device) for k, v in x1.items()}
            inputs['output_hidden_states'] = True
            outputs = model(**inputs)
            
            # Get last hidden state, remove CLS token (first token)
            hidden_states = outputs.hidden_states[-1][:, 1:, :].detach().cpu().numpy()
            
            for x in hidden_states:
                if args.output_embed_format == 'all':
                    embed_array = x  # Shape: (seq_len, hidden_size)
                elif args.output_embed_format == 'ave':
                    embed_array = np.mean(x, axis=0)  # Shape: (hidden_size,)
                
                res_list.append(embed_array)
                
                # Save to file if reached max number
                if len(res_list) >= args.max_seq_num_in_output_file:
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_pkl = os.path.join(args.output_dir, f'embed_{count_file_num}.pkl')
                    with open(output_pkl, 'wb') as f:
                        pickle.dump(res_list, f)
                    print(f"Saved {len(res_list)} embeddings to {output_pkl}")
                    res_list = []
                    count_file_num += 1
    
    # Save remaining embeddings
    if len(res_list) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_pkl = os.path.join(args.output_dir, f'embed_{count_file_num}.pkl')
        with open(output_pkl, 'wb') as f:
            pickle.dump(res_list, f)
        print(f"Saved {len(res_list)} embeddings to {output_pkl}")
    
    tick2 = time.time()
    print("=" * 60)
    print('Done!')
    print(f'Time used: {tick2 - tick1:.2f} seconds')
    print(f'Total sequences processed: {line_num}')
    print("=" * 60)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
