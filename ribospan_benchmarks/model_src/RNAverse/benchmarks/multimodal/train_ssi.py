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
from transformers.trainer_utils import IntervalStrategy
import torch
import transformers
import sklearn
import scipy
import numpy as np
from torch.utils.data import Dataset

from rnaverse.configuration_rnaverse import RNAverseConfig
from rnaverse.modeling_rnaverse import RNAverseForStructuralimputation, RNAverseSeqForStructuralimputation
from rnaverse.tokenization_rnaverse import RNAverseTokenizer
from rnaverse.utils import tokenize_struct_score

early_stopping = EarlyStoppingCallback(early_stopping_patience=20)

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
    evaluation_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    save_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    eval_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate_lm: float = field(default=1e-5, metadata={"help": "Learning rate for the language model part."})
    learning_rate_head: float = field(default=1e-3, metadata={"help": "Learning rate for the classification head."})
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
    metric_for_best_model : str = field(default="r^2")
    greater_is_better: bool = field(default=True)
    stage: str = field(default='0')
    model_type: str = field(default='rnaverse')
    train_from_scratch: bool = field(default=False)
    fusion_mode: str = field(default='late', metadata={"help": "early or late, default late"})
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
        
        if len(data[0]) == 3:
            # data is in the format of [text, struct, label]
            texts = [d[0].upper() for d in data]
            # struct score are stored in a string separated by space
            struct = np.array([list(map(float, d[1].split())) for d in data]).astype(np.float32)
            # labels are stored in a string separated by space
            labels = np.array([list(map(float, d[2].split())) for d in data]).astype(np.float32)
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        seq_length = len(texts[0])
        
        # ensure tokenizer
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        input_dict = {'seq': texts[0]}
        print(tokenizer([input_dict]))
        
        input_dict_list = []
        for i, x in enumerate(texts):
            input_dict = {}
            input_dict['seq'] = x
            input_dict['probe_method'] = 'icSHAPE'
            input_dict['struct_score_list'] = struct[i,:].tolist()
            input_dict_list.append(input_dict)
        
        output = tokenizer(
            input_dict_list, padding=True, max_length=args.model_max_length, truncation=True, return_tensors='pt'
        )
        self.input_ids = output["input_ids"]
        self.input_struct_tokens = output["input_struct_tokens"]
        self.input_probe_method_tokens = output["input_probe_method_tokens"]
        self.input_func_tokens = output["input_func_tokens"]
        self.attention_mask = output["attention_mask"]
        self.weight_mask = torch.ones((self.input_ids.shape[0], seq_length + 1))
        self.post_token_length = torch.zeros(self.attention_mask.shape)

        self.labels = labels
        self.struct = struct
        self.num_labels = 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.attention_mask[i], struct=self.struct[i],
            weight_mask=self.weight_mask[i], post_token_length=self.post_token_length[i],
            input_struct_tokens=self.input_struct_tokens[i],
            input_probe_method_tokens=self.input_probe_method_tokens[i],
            input_func_tokens=self.input_func_tokens[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, struct, weight_mask, post_token_length, input_struct_tokens, input_probe_method_tokens, input_func_tokens = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask", "struct","weight_mask","post_token_length","input_struct_tokens","input_probe_method_tokens","input_func_tokens"))
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        labels = torch.tensor(np.array(labels))
        struct = torch.tensor(np.array(struct))
        input_struct_tokens = torch.stack(input_struct_tokens)
        input_probe_method_tokens = torch.stack(input_probe_method_tokens)
        input_func_tokens = torch.stack(input_func_tokens)
        # struct == -1 and labels != -1 are the positions to predict
        label_mask = (struct == -1) & (labels != -1)
        # 11-way bin the struct score into token (0-9 for 10 bins, 10 for null/masked value)
        struct_token = tokenize_struct_score(struct, bin_num=10, null_value=-1)
        return dict(
            input_ids=input_ids,
            labels=labels[label_mask],
            attention_mask=attention_mask,
            struct=struct_token,
            weight_mask=weight_mask,
            post_token_length=post_token_length,
            l_mask=label_mask,
            input_struct_tokens=input_struct_tokens,
            input_probe_method_tokens=input_probe_method_tokens,
            input_func_tokens=input_func_tokens,
        )

"""
Manually calculate the mse and r^2.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze()
    logits = logits.squeeze()
    return {
        "r^2" : scipy.stats.pearsonr(labels, logits)[0]**2,
        "mse": sklearn.metrics.mean_squared_error(labels, logits),
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
    test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # Load RNAverse model
    is_seq_only = (training_args.model_type == 'rnaverse-seq')
    
    if training_args.train_from_scratch:
        print('Train from scratch')
        config = RNAverseConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
            is_seq_only_model=is_seq_only,
        )
        if is_seq_only:
            model = RNAverseSeqForStructuralimputation(config, fusion_mode=training_args.fusion_mode)
        else:
            model = RNAverseForStructuralimputation(config)
    else:
        print(f'Loading {training_args.model_type} model')
        if is_seq_only:
            model = RNAverseSeqForStructuralimputation.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                problem_type="regression",
                fusion_mode=training_args.fusion_mode,
            )
        else:
            model = RNAverseForStructuralimputation.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                problem_type="regression",
            )

    # define optimizer with two learning rates
    training_args.ddp_find_unused_parameters = True
    training_args.save_model = True
    training_args.label_names = ["labels"]
    lm_para = []
    head_para = []
    for name, param in model.named_parameters():
        if name.startswith('struct_lm_head') or \
                name.startswith('struct_embed') \
                or name.startswith('dense') \
                or name.startswith('classifier') \
                or name.startswith('transfer') \
                or name.startswith('embedding_struct') \
                or name.startswith('LayerNorm'):
            head_para.append(param)
        else:
            lm_para.append(param)
    optimzer_grouped_parameters = [
        {'params': lm_para, 'lr': training_args.learning_rate_lm},
        {'params': head_para, 'lr': training_args.learning_rate_head},
    ]
    optimizer = torch.optim.AdamW(optimzer_grouped_parameters, lr=training_args.learning_rate_head, weight_decay=training_args.weight_decay)
    import math
    num_training_steps = (
        math.ceil(
            math.ceil(len(train_dataset) / training_args.per_device_train_batch_size)
            / training_args.gradient_accumulation_steps
        )
        * training_args.num_train_epochs
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   optimizers=(optimizer, scheduler),
                                   callbacks=[early_stopping],
                                   )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        
        os.makedirs(results_path, exist_ok=True)
        results_test = trainer.evaluate(eval_dataset=test_dataset)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results_test, f, indent=4)
        results_val = trainer.evaluate(eval_dataset=val_dataset)
        with open(os.path.join(results_path, "val_results.json"), "w") as f:
            json.dump(results_val, f, indent=4)
        
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
