import warnings
# warnings.filterwarnings("ignore")
import os
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import get_cosine_schedule_with_warmup

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from rnaverse.tasks.ss_data import SSDataset
from rnaverse.tasks.ss_predictor import SSCNNPredictor
from rnaverse.configuration_rnaverse import RNAverseConfig
from rnaverse.modeling_rnaverse import RNAverseModel
from rnaverse.tokenization_rnaverse import RNAverseTokenizer
import scipy
import random
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed ,seed = {args.seed}")

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze().astype(int)
    logits = logits.squeeze()
    #binary classification
    # Apply the sigmoid function to the logits to get the predicted probabilities
    probs = scipy.special.expit(logits) 
    precision = precision_score(labels, probs > 0.5, average='binary')
    recall = recall_score(labels, probs > 0.5,  average='binary')
    f1 = f1_score(labels, probs > 0.5,  average='binary')
    return {
    "precision": precision,
    "recall" : recall,
    "f1" : f1
    }

class collator():
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        seqs = [x['seq'] for x in batch]
        struct = [x['struct'] for x in batch]
        input_dict_list = []
        for x in batch:
            input_dict_list.append({'seq': x['seq']})

        max_len = max([len(seq) for seq in seqs])

        # RNAverse only has CLS token, no SEP token
        weight_mask = torch.ones((len(seqs), max_len + 1))

        data_dict = self.tokenizer(input_dict_list, 
                        padding=True,
                        max_length=self.args.model_max_length, 
                        truncation=True, 
                        return_tensors='pt')
        post_token_length = torch.zeros(data_dict['attention_mask'].shape)

        # each element in struct is a square matrix, but with different shape. We need to pad them to make them the same shape
        struct = np.array([np.pad(x, ((0,max_len-x.shape[0]),(0,max_len-x.shape[1])), 'constant', constant_values=-1) for x in struct])
        struct = torch.tensor(struct)

        data_dict['struct'] = struct
        data_dict['weight_mask'] = weight_mask
        data_dict['post_token_length'] = post_token_length

        return data_dict

def test(model, test_loader, accelerator):
    model.eval()  # Set the model to evaluation mode
    outputs_list = []
    targets_list = []
    with torch.no_grad(): 
        for data_dict in tqdm(test_loader):
            for key in data_dict:
                data_dict[key] = data_dict[key].to(accelerator.device)

            logits = model(data_dict)[:,1:,1:]
            labels = data_dict['struct']
            label_mask = labels != -1
            outputs_list.append(logits.detach().cpu().numpy().reshape(-1,1))
            targets_list.append(labels.detach().cpu().numpy().reshape(-1,1))
        logits = np.concatenate(outputs_list, axis=0)
        labels = np.concatenate(targets_list, axis=0)
    
    metrics = calculate_metric_with_sklearn(logits, labels)
    
    print(f'\nTest: Precision: {metrics["precision"]}, Recall: {metrics["recall"]}, F1: {metrics["f1"]}')
    return metrics

def main(args):
    set_seed(args)
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers,
                              gradient_accumulation_steps=args.gradient_accumulation_steps)
    model_name = args.output_dir.split('/')[-1]

    name = f'[RNA_Secondary_Structure_Prediction]{model_name}_' \
           f'{args.model_type}_' \
           f'lr{args.lr}_' \
           f'bs{args.per_device_train_batch_size}*gs{args.gradient_accumulation_steps}*gpu{accelerator.state.num_processes}_' \
           f'epo{args.num_epochs}_' \
           f'warm{args.warmup_epos}epoch' \
           f'seed{args.seed}'

    if args.is_freeze:
        name += '_freeze'

    model_config = extractor.config
    model = SSCNNPredictor(args, extractor, model_config, tokenizer, args.is_freeze)
    num_params = count_parameters(model)
    print(f"model params are: {num_params}")

    train_dataset = SSDataset(data_path=args.data_path, tokenizer=tokenizer, args=args, mode='train')
    val_dataset = SSDataset(data_path=args.data_path, tokenizer=tokenizer, args=args, mode='val')
    test_dataset = SSDataset(data_path=args.data_path, tokenizer=tokenizer, args=args, mode='test')

    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')
    collate_fn = collator(tokenizer, args)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## num_processes from accelerator
    per_steps_one_epoch = len(
        train_dataset) // args.per_device_train_batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    num_warmup_steps = per_steps_one_epoch * args.warmup_epos
    num_training_steps = per_steps_one_epoch * args.num_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    criterion = nn.BCEWithLogitsLoss()

    train_loss_list = []
    last_val, best_val, best_test = 0, 0, {}
    patience = args.patience
    early_stop_flag = 0
    for epoch in range(args.num_epochs):
        model.train()
        start_time = time.time()
        for data_dict in tqdm(train_loader):
            with accelerator.accumulate(model):
                logits = model(data_dict)[:,1:,1:]
                labels = data_dict['struct']
                label_mask = labels != -1 
                loss = criterion(logits[label_mask].reshape(-1,1), labels[label_mask].reshape(-1,1))
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                lr_scheduler.step()

            gather_loss = accelerator.gather(loss.detach().float()).mean().item()
            train_loss_list.append(gather_loss)

        print(f"epoch {epoch}:")
        val_metrics = test(model, val_loader, accelerator)
    
        if best_val < val_metrics["f1"]:
            best_val = val_metrics["f1"] 
            test_metrics = {}
            print(f"epoch {epoch}:")
            test_metrics = test(model, test_loader, accelerator)
            best_test = test_metrics
        if last_val < val_metrics["f1"]:
            early_stop_flag = 0 
        else:
            early_stop_flag +=1

        if early_stop_flag >= patience:
            print(f"Early stopping")
            break
        last_val = val_metrics["f1"] 
        
        end_time = time.time()
        if accelerator.is_main_process:
            print(
                f'epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}, train_loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.2f}')
           
            log_dict = {'lr': optimizer.param_groups[0]["lr"], 'train_loss': np.mean(train_loss_list)}
            log_dict.update(val_metrics)
            
            log_dict.update(best_test)
        torch.cuda.empty_cache()
        train_loss_list = []

    results_path = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, f"test_results.json"), "w") as f:
        json.dump(best_test, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epos', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument("--pretrained_lm_dir", type=str, default='')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--model_name_or_path', default='output')
    parser.add_argument('--output_dir', default='./ckpts/')
    parser.add_argument('--model_type', type=str, default='rnaverse')
    parser.add_argument('--model_max_length', type=int, default=1026)
    parser.add_argument('--run_name', type=str, default="run")
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--data_train_path', default='')
    parser.add_argument('--data_val_path', default='')
    parser.add_argument('--data_test_path', default='')
    args = parser.parse_args()

    assert args.mode in ['bprna', 'pdb']

    ## Load RNAverse model and tokenizer
    is_seq_only = (args.model_type == 'rnaverse-seq')
    config = RNAverseConfig.from_pretrained(
        args.pretrained_lm_dir if args.pretrained_lm_dir else args.model_name_or_path,
        is_seq_only_model=is_seq_only,
    )
    tokenizer = RNAverseTokenizer.from_pretrained(
        args.pretrained_lm_dir if args.pretrained_lm_dir else args.model_name_or_path,
        config=config,
    )
    extractor = RNAverseModel.from_pretrained(
        args.pretrained_lm_dir if args.pretrained_lm_dir else args.model_name_or_path,
        config=config,
    )

    main(args)
