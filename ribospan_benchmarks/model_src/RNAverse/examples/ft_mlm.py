import argparse
import torch
import time
import os
import numpy as np
from transformers import set_seed
from rnaverse import RNAverseForMaskedLM, RNAverseTokenizer, RNAverseConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='''
    Fine-tune RNAverse on a small set of RNA sequences using MLM task.
    This script supports fine-tuning on new function families/keywords.
    The config and tokenizer can be customized (e.g., with new num_func_tokens),
    while the model weights are loaded from a pretrained checkpoint.
    ''')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=640,
                        help='batch size for training, validation and test, default is 640')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, default is 1e-4')
    parser.add_argument('--pad_len', type=int, default=512,
                        help='maximum sequence length after padding (including CLS token), default is 512')
    parser.add_argument('--random_seed', type=int, default=37,
                        help='random seed, default is 37')

    # model paths - config and tokenizer may be customized for new function families
    parser.add_argument('--model_path', type=str, default=None,
                        help='pretrained model path for loading weights')
    parser.add_argument('--config_path', type=str, default=None,
                        help='config path (may have customized num_func_tokens for new families)')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='tokenizer path (may have customized func_vocab for new families)')

    # save and resume
    parser.add_argument('--save_model_dir', type=str, default=None,
                        help='directory to save the best model')

    # training control
    parser.add_argument('--max_epoch', type=int, default=30,
                        help='maximum number of epochs, default is 30')
    parser.add_argument('--tol_epoch', type=int, default=5,
                        help='early stopping patience, default is 5')

    # data files
    parser.add_argument('--train_txt', type=str, default=None,
                        help='training data file (tab-separated: name, anno_str, seq)')
    parser.add_argument('--val_txt', type=str, default=None,
                        help='validation data file')
    parser.add_argument('--test_txt', type=str, default=None,
                        help='test data file')

    # fine-tuning mode
    parser.add_argument('--whether_ft', type=int, default=0,
                        help='whether to fine-tune the whole model (1) or only lm_head (0), default is 0')

    args = parser.parse_args()
    return args

def betalinear30():
    '''
    Samples from a beta linear distribution for mask probability.
    80% chance: sample from Beta(3, 9) distribution
    20% chance: sample from Uniform(0, 1) distribution
    '''
    r1 = np.random.random()
    if r1 > 0.8:
        # Linear (Uniform)
        return np.random.random()
    else:
        return np.random.beta(a=3, b=9, size=1)[0]


class FtDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        data_list = []
        with open(data_file, 'r') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 3:
                    name_, anno_str, seq_ = parts
                    data_list.append((name_, anno_str, seq_))
                elif len(parts) == 2:
                    # support format: anno_str, seq
                    anno_str, seq_ = parts
                    data_list.append(('', anno_str, seq_))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        name_, anno_str, seq_ = self.data_list[idx]
        return name_, anno_str, seq_


def collate_fn_mlm(batch_data, tokenizer, pad_len, is_train=True):
    '''
    Collate function for MLM training using RNAverseTokenizer.
    
    Args:
        batch_data: list of (name, anno_str, seq) tuples
        tokenizer: RNAverseTokenizer instance
        pad_len: maximum sequence length
        is_train: if True, use betalinear30 for mask probability; if False, use fixed 0.3
    
    Returns:
        dict with input tensors and labels
    '''
    raw_str_batch = []
    gt_seqs = []
    
    for name_, anno_str, seq_ in batch_data:
        # Truncate sequence if too long
        if len(seq_) > pad_len - 1:  # -1 for CLS token
            random_start = np.random.randint(0, len(seq_) - pad_len + 1)
            seq_ = seq_[random_start:random_start + pad_len - 1]
        
        # Parse function annotations
        func_list = anno_str.split(',') if anno_str else ['UNK']
        
        # Determine mask probability
        if is_train:
            mask_prob = betalinear30()
        else:
            mask_prob = 0.3
        
        # Randomly select positions to mask
        # mask_pos_list = []
        # for i in range(len(seq_)):
        #     if np.random.random() < mask_prob:
        #         mask_pos_list.append(i)
        mask_pos_num = max(1, int(len(seq_) * mask_prob))
        mask_pos_list = np.random.choice(len(seq_), size=mask_pos_num, replace=False).tolist()
        
        raw_str_batch.append({
            'seq': seq_,
            'func_list': func_list,
            'mask_pos_list': mask_pos_list,
        })
        gt_seqs.append(seq_)
    
    # Use tokenizer to prepare inputs
    batch_inputs = tokenizer(raw_str_batch, padding=True, truncation=False, return_tensors='pt')
    
    # Prepare ground truth labels
    dict_base_id = tokenizer.dict_base_id
    gt_ids = []
    for seq_ in gt_seqs:
        seq_ids = [dict_base_id['CLS']] + [dict_base_id.get(c, dict_base_id.get('UNK', 0)) for c in seq_]
        # Pad to max length
        max_len = batch_inputs['input_ids'].shape[1]
        seq_ids = seq_ids + [dict_base_id['PAD']] * (max_len - len(seq_ids))
        gt_ids.append(seq_ids)
    
    batch_inputs['labels'] = torch.tensor(gt_ids, dtype=torch.long)
    
    # Create mask indicator (1 for masked positions)
    mask_indicator = (batch_inputs['input_ids'] == dict_base_id['MASK']).long()
    batch_inputs['mask_indicator'] = mask_indicator
    
    return batch_inputs


def loss_fn_mlm(output, label, new_id_mask):
    '''
    Compute MLM loss only on masked positions.
    '''
    # output shape is bs* seq_len+1 * vocab_size
    # label shape is bs*seq_len+1
    output = output.permute(0,2,1)
    loss_ce = torch.nn.CrossEntropyLoss(reduction='none')
    if torch.sum(new_id_mask) == 0:
        return torch.sum(loss_ce(output, label)*new_id_mask)
    else:
        loss = torch.sum(loss_ce(output, label)*new_id_mask)/torch.sum(new_id_mask)
    return loss


def compute_acc(logits, labels, mask_indicator):
    '''
    Compute accuracy on masked positions.
    '''
    pred = torch.argmax(logits, dim=-1)
    correct = ((pred == labels) * mask_indicator).sum()
    total = mask_indicator.sum()
    return correct, total


def validate(model, dataloader, device):
    model.eval()
    val_loss_list = []
    val_loss_weight_list = []
    val_acc = 0
    val_total = 0

    for batch_inputs in tqdm(dataloader, desc='Validating'):
        with torch.no_grad():
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            outputs = model(
                input_ids=batch_inputs['input_ids'],
                input_struct_tokens=batch_inputs.get('input_struct_tokens'),
                input_probe_method_tokens=batch_inputs.get('input_probe_method_tokens'),
                input_func_tokens=batch_inputs.get('input_func_tokens'),
                attention_mask=batch_inputs['attention_mask'],
            )
            
            loss = loss_fn_mlm(outputs.logits, batch_inputs['labels'], batch_inputs['mask_indicator'])
            acc, total = compute_acc(outputs.logits, batch_inputs['labels'], batch_inputs['mask_indicator'])
            
            val_loss_list.append(loss.item())
            val_loss_weight_list.append(batch_inputs['mask_indicator'].sum().item())
            val_acc += acc.item()
            val_total += total.item()

    if val_total == 0:
        return 0.0, 0.0, 0
    
    val_loss = np.sum(np.array(val_loss_list) * np.array(val_loss_weight_list)) / np.sum(val_loss_weight_list)
    return val_loss, val_acc, val_total


def main(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    set_seed(args.random_seed)

    tick1 = time.time()

    # Load tokenizer
    if args.tokenizer_path is not None:
        print(f'Loading tokenizer from {args.tokenizer_path}')
        tokenizer = RNAverseTokenizer.from_pretrained(args.tokenizer_path)
    else:
        raise ValueError('tokenizer_path is required')

    # Load config
    if args.config_path is not None:
        print(f'Loading config from {args.config_path}')
        config = RNAverseConfig.from_pretrained(args.config_path)
    else:
        print('Using default config')
        config = RNAverseConfig()

    # Load model
    if args.model_path is not None:
        print(f'Loading pretrained model from {args.model_path}')
        
        pretrained_state_path = os.path.join(args.model_path, 'pytorch_model.bin')
        if os.path.exists(pretrained_state_path):
            pretrained_state = torch.load(pretrained_state_path, map_location='cpu')
        else:
            pretrained_state = None
        
        model = RNAverseForMaskedLM(config)
        
        if pretrained_state is not None:
            func_embed_key = 'rnaverse.embeddings.func_embeddings.weight'
            
            if func_embed_key in pretrained_state:
                pretrained_weight = pretrained_state[func_embed_key]
                current_weight = model.state_dict()[func_embed_key]
                
                if pretrained_weight.shape != current_weight.shape:
                    print(f"Aligning {func_embed_key}: {pretrained_weight.shape} -> {current_weight.shape}")
                    
                    new_weight = current_weight.clone() 
                    
                    rows_to_copy = min(pretrained_weight.shape[0], current_weight.shape[0])
                    cols_to_copy = min(pretrained_weight.shape[1], current_weight.shape[1])
                    
                    new_weight[:rows_to_copy, :cols_to_copy] = pretrained_weight[:rows_to_copy, :cols_to_copy]
                    
                    pretrained_state[func_embed_key] = new_weight.to(pretrained_weight.dtype)

            missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
            
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
                
            print('Pretrained weights loaded successfully')
    else:
        print('No pretrained model provided, using random initialization')
        model = RNAverseForMaskedLM(config)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    
    # Freeze parameters if not fine-tuning the whole model
    if args.whether_ft == 0:
        for name, param in model.named_parameters():
            if name.startswith('lm_head'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print('Only lm_head is trainable')
    else:
        print('All parameters are trainable')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load datasets
    train_dataset = FtDataset(args.train_txt)
    # val_dataset = FtDataset(args.val_txt)
    if args.val_txt is not None:
        val_dataset = FtDataset(args.val_txt)
    else:
        '''
        If no separate val set, use test set for validation.
        It is the case for IGFBP3 aptamer fine-tuning, where to ensure a fair comparison,
        we followed the same training and test split protocol used in AptaDiff and RNAGenesis
        '''
        val_dataset = FtDataset(args.test_txt)  

    test_dataset = FtDataset(args.test_txt)
    
    print(f'Number of training data: {len(train_dataset)}')
    print(f'Number of validation data: {len(val_dataset)}')
    print(f'Number of test data: {len(test_dataset)}')

    # Create dataloaders with collate_fn
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_mlm(batch, tokenizer, args.pad_len, is_train=True),
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mlm(batch, tokenizer, args.pad_len, is_train=False)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mlm(batch, tokenizer, args.pad_len, is_train=False)
    )

    # Evaluate before training
    print('\n' + '='*20 + ' Before Training ' + '='*20)
    train_loss, train_acc, train_total = validate(model, train_dataloader, device)
    val_loss, val_acc, val_total = validate(model, val_dataloader, device)
    test_loss, test_acc, test_total = validate(model, test_dataloader, device)
    
    print(f'Train loss: {train_loss:.4f}, acc: {train_acc/train_total:.4f} ({train_acc}/{train_total})')
    print(f'Val loss: {val_loss:.4f}, acc: {val_acc/val_total:.4f} ({val_acc}/{val_total})')
    print(f'Test loss: {test_loss:.4f}, acc: {test_acc/test_total:.4f} ({test_acc}/{test_total})')

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    best_metrics = None
    best_model_state_dict = None

    for epoch in range(args.max_epoch):
        print('\n' + '='*20 + f' Epoch {epoch} ' + '='*20)
        model.train()
        
        train_loss_list = []
        train_loss_weight_list = []
        train_acc = 0
        train_total = 0
        
        for batch_inputs in tqdm(train_dataloader, desc='Training'):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            outputs = model(
                input_ids=batch_inputs['input_ids'],
                input_struct_tokens=batch_inputs.get('input_struct_tokens'),
                input_probe_method_tokens=batch_inputs.get('input_probe_method_tokens'),
                input_func_tokens=batch_inputs.get('input_func_tokens'),
                attention_mask=batch_inputs['attention_mask'],
            )
            
            loss = loss_fn_mlm(outputs.logits, batch_inputs['labels'], batch_inputs['mask_indicator'])
            acc, total = compute_acc(outputs.logits, batch_inputs['labels'], batch_inputs['mask_indicator'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())
            train_loss_weight_list.append(batch_inputs['mask_indicator'].sum().item())
            train_acc += acc.item()
            train_total += total.item()
        
        train_loss = np.sum(np.array(train_loss_list) * np.array(train_loss_weight_list)) / np.sum(train_loss_weight_list)
        print(f'Train loss: {train_loss:.4f}, acc: {train_acc/train_total:.4f} ({train_acc}/{train_total})')
        
        # Validation
        val_loss, val_acc, val_total = validate(model, val_dataloader, device)
        print(f'Val loss: {val_loss:.4f}, acc: {val_acc/val_total:.4f} ({val_acc}/{val_total})')
        
        # Test
        test_loss, test_acc, test_total = validate(model, test_dataloader, device)
        print(f'Test loss: {test_loss:.4f}, acc: {test_acc/test_total:.4f} ({test_acc}/{test_total})')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = [train_loss, train_acc, train_total,
                           val_loss, val_acc, val_total,
                           test_loss, test_acc, test_total]
            
            if args.save_model_dir is not None:
                os.makedirs(args.save_model_dir, exist_ok=True)
                # Save model, config, and tokenizer using save_pretrained
                model.save_pretrained(args.save_model_dir)
                tokenizer.save_pretrained(args.save_model_dir)
                config.save_pretrained(args.save_model_dir)
                print(f'Best model, config, and tokenizer saved to {args.save_model_dir}')
        else:
            if epoch - best_epoch >= args.tol_epoch:
                print(f'Early stopping at epoch {epoch}')
                break

    # Print best results
    print('\n' + '='*20 + ' Best Results ' + '='*20)
    print(f'Best epoch: {best_epoch}')
    print(f'Train loss: {best_metrics[0]:.4f}, acc: {best_metrics[1]/best_metrics[2]:.4f}')
    print(f'Val loss: {best_metrics[3]:.4f}, acc: {best_metrics[4]/best_metrics[5]:.4f}')
    print(f'Test loss: {best_metrics[6]:.4f}, acc: {best_metrics[7]/best_metrics[8]:.4f}')

    tick2 = time.time()
    print(f'\nTime cost: {tick2 - tick1:.2f} seconds')
    print('Done!')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
