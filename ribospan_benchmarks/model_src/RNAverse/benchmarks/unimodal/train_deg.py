import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers.trainer_utils import IntervalStrategy
import random

import torch
import transformers
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from rnaverse.configuration_rnaverse import RNAverseConfig
from rnaverse.modeling_rnaverse import RNAverseForNucleotideLevel
from rnaverse.tokenization_rnaverse import RNAverseTokenizer

# early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    tokenizer_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1025, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    # evaluation_strategy: str = field(default="steps")
    evaluation_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    save_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    eval_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=True)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    report_to: str = field(default=None)
    metric_for_best_model : str = field(default="mcrmse")
    greater_is_better: bool = field(default=False)
    stage: str = field(default='0')
    model_type: str = field(default='rnaverse')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    remove_checkpoint_folders: bool = field(default=True, metadata={"help": "Whether to remove checkpoint folders after training."})


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed ,seed = {args.seed}")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class SupervisedDataset(Dataset):

    def __init__(self, data_path, tokenizer, signal_noise_cutoff, test_set=None, args=None):
        super().__init__()
        self.df = pd.read_json(data_path)
        print('pre',self.df.shape)
        deg_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        
        self.is_test = test_set is not None or deg_cols[0] not in self.df.columns
        if self.is_test:
            self.df = self.df.query(("seq_length == 107" if test_set == 'public' else "seq_length == 130"))
            self.y = None
        else:
            self.df = self.df[self.df.signal_to_noise >= signal_noise_cutoff]
            self.y = np.stack([np.stack(self.df[col].values) for col in deg_cols], axis=-1)
        print('post',self.df.shape)
        self.sample_ids = self.df['id'].values
        texts = [d.upper() for d in self.df['sequence']]
               
        seq_length = len(texts[0])
        
        # ensure tokenier
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        input_dict_test = {'seq': texts[0]}
        print(tokenizer([input_dict_test]))

        input_dict_list = [{'seq': seq} for seq in texts]
        output = tokenizer(
            input_dict_list, padding=True, max_length=args.model_max_length, truncation=True, return_tensors='pt'
        )
        self.input_ids = output["input_ids"]
        self.input_struct_tokens = output["input_struct_tokens"]
        self.input_probe_method_tokens = output["input_probe_method_tokens"]
        self.input_func_tokens = output["input_func_tokens"]
        self.texts = texts
        #make sure the length of sequences in the dataset is the same
        self.weight_mask = torch.ones((self.input_ids.shape[0],seq_length+1)) # +1 for [cls]

        self.attention_mask = output["attention_mask"]
        self.post_token_length = torch.zeros(self.attention_mask.shape)
        self.num_labels = 3

    def __getitem__(self, index: int):
        if self.is_test:          
            sample_id = self.sample_ids[index]
            return dict(input_ids=self.input_ids[index], sample_ids=sample_id, attention_mask=self.attention_mask[index],
                weight_mask=self.weight_mask[index],post_token_length=self.post_token_length[index],
                input_struct_tokens=self.input_struct_tokens[index],
                input_probe_method_tokens=self.input_probe_method_tokens[index],
                input_func_tokens=self.input_func_tokens[index])
        targets = torch.tensor(self.y[index, :, :], dtype=torch.float32)
        return dict(input_ids=self.input_ids[index], labels=targets, attention_mask=self.attention_mask[index],
           weight_mask=self.weight_mask[index],post_token_length=self.post_token_length[index],
           input_struct_tokens=self.input_struct_tokens[index],
           input_probe_method_tokens=self.input_probe_method_tokens[index],
           input_func_tokens=self.input_func_tokens[index])
     
    
    def __len__(self) -> int:
        return self.df.shape[0]

@dataclass
class TestDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, sample_ids, attention_mask, weight_mask, post_token_length, input_struct_tokens, input_probe_method_tokens, input_func_tokens = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"sample_ids", "attention_mask","weight_mask","post_token_length","input_struct_tokens","input_probe_method_tokens","input_func_tokens"))
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        input_struct_tokens = torch.stack(input_struct_tokens)
        input_probe_method_tokens = torch.stack(input_probe_method_tokens)
        input_func_tokens = torch.stack(input_func_tokens)
        return dict(
            input_ids=input_ids,
            sample_ids=sample_ids,
            attention_mask=attention_mask,
            weight_mask=weight_mask,
            post_token_length=post_token_length,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, weight_mask, post_token_length, input_struct_tokens, input_probe_method_tokens, input_func_tokens = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask","weight_mask","post_token_length","input_struct_tokens","input_probe_method_tokens","input_func_tokens"))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        input_struct_tokens = torch.stack(input_struct_tokens)
        input_probe_method_tokens = torch.stack(input_probe_method_tokens)
        input_func_tokens = torch.stack(input_func_tokens)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            weight_mask=weight_mask,
            post_token_length=post_token_length,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
        )

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    def rmse(labels,logits):
        return np.mean(np.square(labels - logits + 1e-6))
    score = 0
    num_scored = 3
    for i in range(num_scored):
        score += rmse(labels[:, :, i], logits[:, :, i]) / num_scored       
    return {
        "mcrmse": score
    }
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)

def build_submission_df(ids, pred_tensor):
    if type(pred_tensor).__module__ != np.__name__:
        pred_tensor = pred_tensor.cpu().detach().numpy()
    res = []
    for i, id in enumerate(ids):
        
        for j, pred in enumerate(pred_tensor[i, :, :]):
            res.append([id+'_'+str(j)] + list(pred))
    return res

def make_pred_file(args, model, loaders, postfix=''):
    res = []
    model.to(args.device)
    print(args.device)
    model.eval()
    for eval_dataloader in loaders:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            sample_ids = batch["sample_ids"]
            weight_mask = batch["weight_mask"].to(args.device)
            post_token_length = batch["post_token_length"].to(args.device)
            input_struct_tokens = batch["input_struct_tokens"].to(args.device)
            input_probe_method_tokens = batch["input_probe_method_tokens"].to(args.device)
            input_func_tokens = batch["input_func_tokens"].to(args.device)
            with torch.no_grad():
                test_pred = model(input_ids=input_ids, attention_mask=attention_mask, weight_mask=weight_mask, post_token_length=post_token_length,
                                  input_struct_tokens=input_struct_tokens,
                                  input_probe_method_tokens=input_probe_method_tokens,
                                  input_func_tokens=input_func_tokens)
                test_pred = test_pred[0][:, 1:,:] #exclude [cls] and [sep]
                res += build_submission_df(sample_ids, test_pred)

    pred_df = pd.DataFrame(res, columns=['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])
    pred_df['deg_pH10'] = 0
    pred_df['deg_50C'] = 0
    results_path = os.path.join(args.output_dir,"results", args.run_name)
    print(results_path)
    os.makedirs(results_path, exist_ok=True)
    results_path = os.path.join(results_path, 'submission_'+postfix+'.csv')
    pred_df.to_csv(results_path, index=False)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    
    # Load RNAverse tokenizer
    tokenizer = RNAverseTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
    )

    train_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_train_path), tokenizer, signal_noise_cutoff=0.6, test_set=None, args=training_args)
    val_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_val_path), tokenizer, signal_noise_cutoff=1.0, test_set=None, args=training_args)
    public_test_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_test_path), tokenizer, signal_noise_cutoff=-99.0, test_set='public', args=training_args)
    private_test_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_test_path), tokenizer, signal_noise_cutoff=-99.0, test_set='private', args=training_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    test_data_collator = TestDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(private_test_dataset)}+{len(private_test_dataset)}')

    # Load RNAverse model
    if training_args.train_from_scratch:
        print('Train from scratch')
        config = RNAverseConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
        )
        model = RNAverseForNucleotideLevel(config)
    else:
        print(f'Loading {training_args.model_type} model')
        print(train_dataset.num_labels)
        model = RNAverseForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
        )
       
    training_args.ddp_find_unused_parameters = True
    training_args.save_model = True
    early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[early_stopping],
                                   )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
   
   
    def test_dataset_loader(dataset, args):
        return DataLoader(
            dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=test_data_collator,
            num_workers=4
        )
    test_data_loader1 = test_dataset_loader(public_test_dataset,training_args)
    test_data_loader2 = test_dataset_loader(private_test_dataset,training_args)
    make_pred_file(training_args, model, [test_data_loader1, test_data_loader2],postfix=training_args.output_dir.split('/')[-1])
    
    # remove checkpoint folders if specified
    if training_args.remove_checkpoint_folders:
        for folder_name in os.listdir(training_args.output_dir):
            if folder_name.startswith("checkpoint-"):
                folder_path = os.path.join(training_args.output_dir, folder_name)
                if os.path.isdir(folder_path):
                    print(f"Removing checkpoint folder: {folder_path}")
                    import shutil
                    shutil.rmtree(folder_path)

if __name__ == "__main__":
    train()

#how to get score:
#submit to kaggle with command like
#kaggle competitions submit -c stanford-covid-vaccine -f xxx/submission_yyy.csv -m "Message"
