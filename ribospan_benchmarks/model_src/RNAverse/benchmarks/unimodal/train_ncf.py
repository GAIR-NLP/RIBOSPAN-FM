import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers.trainer_utils import IntervalStrategy
import torch
import random
import sklearn
import transformers

import numpy as np
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

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
    # evaluation_strategy: str = field(default="steps"),
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
    fp16: bool = field(default=False)
    metric_for_best_model: str = field(default="accuracy")
    greater_is_better: bool = field(default=True)
    stage: str = field(default='0')
    model_type: str = field(default='rnaverse')
    report_to: str = field(default=None)
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    remove_checkpoint_folders: bool = field(default=True, metadata={"help": "Whether to remove checkpoint folders after training."})

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.distributed.get_rank() >= 0:
        print("!!!!!!!!!!!!!", "Yes")
        torch.cuda.manual_seed_all(args.seed)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0].upper() for d in data]
            labels = [int(d[1]) for d in data]
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        text = texts[0]
        
        # ensure tokenier
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        input_dict = {'seq': texts[0]}
        print(tokenizer([input_dict]))
        self.labels = labels
        self.num_labels = len(set(labels))
        self.texts = texts

    def __len__(self):
       return len(self.texts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.texts[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        seqs, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_dict_list = [{'seq': seq} for seq in seqs]
        input_dict = self.tokenizer(input_dict_list, padding=True, max_length=self.args.model_max_length, truncation=True, return_tensors='pt')
        labels = torch.Tensor(labels).long()
        input_dict['labels'] = labels
        return input_dict

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        "precision": sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0),
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

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
    test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # Load RNAverse model
    if training_args.train_from_scratch:
        print('Train from scratch')
        config = RNAverseConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
        )
        model = RNAverseForSequenceClassification(config)
    else:
        print(f'Loading {training_args.model_type} model')
        print(train_dataset.num_labels)
        model = RNAverseForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
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

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        print("on the test set:", results, "\n", results_path)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
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
