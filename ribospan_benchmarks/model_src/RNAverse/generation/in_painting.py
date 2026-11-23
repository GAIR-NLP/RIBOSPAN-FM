import argparse
import torch
import numpy as np
import copy
import os
from transformers import set_seed
from rnaverse import RNAverseForMaskedLM, RNAverseTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description='''
    In-painting RNA sequence generator for specific Rfam family.
    If directly use pretrained RNAverse model, it will be Zero-shot in-painting. 
    You can also further fine-tune the model on target sequences before in-painting.
    ''')
    parser.add_argument('--init_seq', type = str, default=None,
                        help = '''
                            initial 2-line format fasta sequence (will read the second line in this file as input sequence),
                            N residues are positions to be generated.
                            ''')
    parser.add_argument('--seq_len', type = int, default=100, help = 'length of the generated sequence. If init_seq is provided, this argument will be ignored.')
    parser.add_argument('--output_fa', type = str, required=True, help = 'output fasta file path')
    # arguments for model
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help = 'Path to pretrained model folder containing model, config, and tokenizer files')
    parser.add_argument('--rfam_id', type = str, default='RF00167', 
                        help = 'Rfam id for functional annotation, default is RF00167')
    # arguments for generator
    parser.add_argument('--random_seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('--seq_num', type = int, default = 100, help = 'number of sequences to generate')
    # generation iteration
    parser.add_argument('--max_token_per_iter', type = int, default = 5, 
                        help = 'max number of tokens to be generated in each iteration.')
    # temperature
    parser.add_argument('--temperature', type = float, default = 1.0, help = 'temperature for sampling')

    args = parser.parse_args()
    return args


def dict_to_device(d, device):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            d[key] = torch.from_numpy(value).to(device)
        elif isinstance(value, bool):
            d[key] = value
        else:
            raise ValueError("Please check your data type!")
    return d

class IterativeGenerator():
    def __init__(self,
                 rfam_id,
                 model_name_or_path,
                 max_iter=10,
                 temperature=1.0,
                 seq_num=100,
    ):
        self.max_iter = max_iter
        self.temperature = temperature
        self.seq_num = seq_num
        self.rfam_id = rfam_id
        
        # Load tokenizer and model from pretrained model folder
        print('-'*10 + f'Loading prediction model ...' + '-'*10)
        self.tokenizer = RNAverseTokenizer.from_pretrained(model_name_or_path)
        self.model = RNAverseForMaskedLM.from_pretrained(model_name_or_path)
        self.model.eval()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    @torch.no_grad()
    def generate(self, init_seq):
        dict_token_seq = {0:'A', 1:'U', 2:'C', 3:'G'}

        # initialize sequence and mask list
        seq_lst = [init_seq]*self.seq_num
        N_positions = [i for i in range(len(init_seq)) if init_seq[i] == 'N']
        mask_lst = [N_positions for _ in range(self.seq_num)]
        fill_count = len(N_positions) # number of positions to be filled
        print(f'Number of positions to be filled: {fill_count}')

        # iterative generation
        for step in range(self.max_iter):
            print('-'*10 + f'Iteration {step+1} T {self.temperature:.1f}' + '-'*10)

            # predict
            logits = self.get_logits(seq_lst, mask_lst) # seq_lst and mask_lst are updated in each iteration
            probs = torch.softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]

            tokens, scores = self.sample_from_categorical(logits)
            prev_seq_lst, prev_mask_lst = copy.deepcopy(seq_lst), copy.deepcopy(mask_lst)

            output_mask = torch.zeros_like(scores)
            for i in range(self.seq_num):
                for j in prev_mask_lst[i]:
                    output_mask[i, j] = 1 # mask position = 1

            output_scores = scores * output_mask # mask out non-masked positions

            # skeptical unmasking
            if step + 1 < self.max_iter:
                p = 1.0 - (step + 1) / self.max_iter
                mask_num = int(fill_count * p)
                mask_lst = self.skeptical_unmasking(output_scores, mask_num)
            else:
                mask_lst = [[] for _ in range(self.seq_num)]
            
            # update seq_lst
            seq_lst = [list(seq) for seq in seq_lst]
            for i in range(self.seq_num):
                for j in prev_mask_lst[i]:
                    if j not in mask_lst[i]:
                        seq_lst[i][j] = dict_token_seq[tokens[i, j].item()]

            seq_lst = [''.join(seq) for seq in seq_lst]
            print(f'Updated sequences example: {seq_lst[0]}')
        
        return seq_lst

    def format_input(self, seq_lst, mask_lst):
        '''
        Prepare input using RNAverseTokenizer
        '''
        raw_str_batch = []
        for i in range(len(seq_lst)):
            seq_ = seq_lst[i]
            mask_pos_list = mask_lst[i] # 0-based
            raw_str_batch.append({
                'seq': seq_,
                'rfam_id': self.rfam_id,
                'mask_pos_list': mask_pos_list,
            })
        
        if self.tokenizer is not None:
            x = self.tokenizer(raw_str_batch, padding=True, return_tensors='pt')
        else:
            raise ValueError('Tokenizer is not provided. Please provide a tokenizer for formatting input.')
        return x
    
    def get_logits(self, seq_lst, mask_lst):
        batch_size = 1000
        logits_lst = []
        for i in range(0, len(seq_lst), batch_size):
            x = self.format_input(seq_lst[i:i+batch_size], mask_lst[i:i+batch_size])
            x = dict_to_device(x, self.device)
            with torch.no_grad():
                outputs = self.model(**x)
                logits = outputs.logits[:,1:, :4].cpu()  # exclude CLS token, only AUCG
                logits_lst.append(logits)
        logits = torch.cat(logits_lst)

        return logits

    def sample_from_categorical(self, logits=None):
        dist = torch.distributions.Categorical(logits=logits.div(self.temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
        return tokens, scores
    
    def skeptical_unmasking(self, output_scores, mask_num):
        # sort the positions by scores
        positions = np.argsort(output_scores, axis=1)[:, :mask_num]
        mask_lst = positions.tolist()
        return mask_lst

def main(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    set_seed(args.random_seed)

    # read input fasta file
    if args.init_seq is None:
        init_seq = 'N'*args.seq_len
    else:
        init_seq = open(args.init_seq).readlines()[1].strip()
    
    max_iter = int(init_seq.count('N')/args.max_token_per_iter) + 1
    generator = IterativeGenerator(rfam_id=args.rfam_id,
                                    model_name_or_path=args.model_name_or_path,
                                    max_iter=max_iter,
                                    temperature=args.temperature,
                                    seq_num=args.seq_num,
    )

    out_dir = os.path.dirname(args.output_fa)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seq_list = generator.generate(init_seq)
    OUT = open(args.output_fa, 'w')
    for i in range(len(seq_list)):
        seq_name = f'seq{i}'
        OUT.write(f'>{seq_name}\n{seq_list[i]}\n')
    OUT.close()
    print(f'Generated sequences are saved to {args.output_fa}.')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    
