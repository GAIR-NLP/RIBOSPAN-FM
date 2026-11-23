import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers.trainer_utils import IntervalStrategy
import random
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

import torch
import transformers
import scipy
import numpy as np
from torch.utils.data import Dataset

from rnaverse.configuration_rnaverse import RNAverseConfig
from rnaverse.modeling_rnaverse import RNAverseForSequenceClassification
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
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: str = field(default=None)
    metric_for_best_model : str = field(default="pearson_r_squared")
    greater_is_better: bool = field(default=True)
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
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedDataset, self).__init__()
        
        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 4:
            # data is in the format of [text, label]
            texts = [d[1].upper() for d in data]
            labels = [float(d[2]) for d in data]
            
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        labels = np.array(labels)
        labels = labels.tolist()
        
        # ensure tokenier
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        input_dict = {'seq': texts[0]}
        print(tokenizer([input_dict]))

        self.texts = texts
        self.labels = labels
        self.num_labels = 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        return dict(input_ids=self.texts[i],labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels"))
        input_dict_list = [{'seq': seq} for seq in input_ids]
        input_dict = self.tokenizer(input_dict_list, padding=True, max_length=self.args.model_max_length, truncation=True, return_tensors='pt')
        labels = torch.Tensor(labels).float()  
        
        input_dict['labels'] = labels
        return input_dict

"""
Manually calculate the mse and r^2.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    # R^2 and pearson_r**2
    # R^2 = 1 - SSE/SST
    labels = labels.squeeze()
    logits = logits.squeeze()
    # log odds to probability
    # clamp the predictions to avoid log(0)
    predictions = np.clip(logits,1e-7,1-1e-7)
    targets = np.clip(labels,1e-7,1-1e-7)
    predictions = np.log(predictions/(1-predictions))
    targets = np.log(targets/(1-targets))
    r2 = 1 -  np.sum((predictions - targets)**2) / np.sum((targets - np.mean(targets))**2)
    pearson_r_squared = scipy.stats.pearsonr(predictions,targets)[0]**2
    return {
    "r^2": r2,
    "pearson_r_squared" : pearson_r_squared,
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    
    # Load RNAverse tokenizer
    tokenizer = RNAverseTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
    )
    
    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path))
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)}')

    # Load RNAverse model
    if training_args.train_from_scratch:
        print('Train from scratch')
        config = RNAverseConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
        )
        model = RNAverseForSequenceClassification(config)
    else:
        print(f'Loading {training_args.model_type} model')
        model = RNAverseForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
        )
    
    # define trainer
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
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    data_test_list = data_args.data_test_path.replace(" ", "").split(",")
    print(f"data_test_list = {len(data_test_list)}")
    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        for data_test in data_test_list:
            data_test_name = data_test +".csv"
            print(f"evaluating data_test_name = {data_test_name}")
            test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                            data_path=os.path.join(data_args.data_path, data_test_name))
            results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
            results = trainer.evaluate(eval_dataset=test_dataset)
            os.makedirs(results_path, exist_ok=True)
            results_test = trainer.evaluate(eval_dataset=test_dataset)
            with open(os.path.join(results_path, f"{data_test}_results.json"), "w") as f:
                json.dump(results_test, f, indent=4)
        
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
