import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers.trainer_utils import IntervalStrategy
import random
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

import torch
import transformers

import numpy as np
from torch.utils.data import Dataset

from rnaverse.configuration_rnaverse import RNAverseConfig
from rnaverse.modeling_rnaverse import RNAverseForSequenceClassificationMeanPool, RNAverseSeqForSequenceClassificationWithStr
from rnaverse.tokenization_rnaverse import RNAverseTokenizer
from rnaverse.utils import tokenize_struct_score

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
    probe_tech: str = field(default='UNK', metadata={"help": "The probing technology used for RNA structure profiling."})

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
    # learning_rate: float = field(default=1e-4)
    learning_rate_lm: float = field(default=1e-5, metadata={"help": "Learning rate for the language model part."})
    learning_rate_head: float = field(default=1e-3, metadata={"help": "Learning rate for the classification head."})
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=True)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    freeze_encoder: bool = field(default=False)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: str = field(default=None)
    metric_for_best_model : str = field(default="auc")
    greater_is_better: bool = field(default=True)
    stage: str = field(default='0')
    model_type: str = field(default='rnaverse')
    train_from_scratch: bool = field(default=False)
    fusion_mode: str = field(default='late', metadata={"help": "early (structure fused into input embeddings) or late (structure added before downstream head), default late"})
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    lr_scheduler_type: str = field(default="linear")
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

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable ratio: {trainable / total:.6f}")

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer,
                 probe_tech: str = 'UNK',
                 ):

        super(SupervisedDataset, self).__init__()
        
        # load data from the disk
        texts = []
        structures = []
        labels = []
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                tmp_list = line.rstrip('\n').split('\t')
                # mod and pas 
                labels.append(float(tmp_list[-1]))
                shape_list = [float(x) for x in tmp_list[-2].split(',')]
                seq_ = tmp_list[-3].upper()
                
                texts.append(seq_)
                structures.append(shape_list)

        labels = np.array(labels)
        labels = labels.tolist()
        
        # ensure tokenizer
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        input_dict = {'seq': texts[0], 
                      'struct_score_list': structures[0], 
                      'probe_method': probe_tech}
        print(tokenizer([input_dict]))
        
        self.texts = texts
        self.structures = structures
        self.labels = labels
        self.num_labels = 1 # 使用"multi_label_classification"以使用BCEWithLogitsLoss

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        return dict(input_ids=self.texts[i],labels=self.labels[i], structures=self.structures[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args, probe_tech):
        self.tokenizer = tokenizer
        self.args = args
        self.probe_tech = probe_tech

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, structures = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "structures"))
        
        input_dict_list = []
        for seq, shape_list in zip(input_ids, structures):
            input_dict_single = {}
            input_dict_single['seq'] = seq
            input_dict_single['struct_score_list'] = shape_list
            input_dict_single['probe_method'] = self.probe_tech
            input_dict_list.append(input_dict_single)
            
        input_dict = self.tokenizer(input_dict_list, padding=True, max_length=self.args.model_max_length, truncation=True, return_tensors='pt')
        
        # here we use BCEWithLogitsLoss
        labels = torch.Tensor(labels).float().unsqueeze(1)
        
        input_dict['labels'] = labels
        if self.args.model_type == 'rnaverse-seq':
            struct_tensor = torch.Tensor(structures).float()
            # 11-way bin the struct score into token (0-9 for 10 bins, 10 for null/masked value)
            struct_token = tokenize_struct_score(struct_tensor, bin_num=10, null_value=-1)
            input_dict['struct'] = struct_token
        return input_dict

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # since is a 2-label classification
    # we want to calculate AUC, accuracy, F1, precision and recall, and auprc
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, average_precision_score
    # logits has not been sigmoid and not between 0 and 1
    probs = torch.sigmoid(torch.Tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    return {
        "auc": roc_auc_score(labels, logits),
        "auprc": average_precision_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    training_args.remove_unused_columns = False
    
    # Load RNAverse tokenizer
    tokenizer = RNAverseTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
    )
    
    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                    data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                    probe_tech=data_args.probe_tech)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                    data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                    probe_tech=data_args.probe_tech)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                    data_path=os.path.join(data_args.data_path, data_args.data_test_path), 
                                    probe_tech=data_args.probe_tech)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args, probe_tech=data_args.probe_tech)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')
    if training_args.eval_steps == -1:
        # evaluate twice in one epoch
        steps_to_use = len(train_dataset) // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps // 2
        training_args.eval_steps = steps_to_use
        training_args.save_steps = steps_to_use
        training_args.logging_steps = steps_to_use
        print(f"eval_steps, save_steps and logging_steps are set to {steps_to_use}.")
    problem_type = "multi_label_classification"
    
    # Load RNAverse model
    is_seq_only = (training_args.model_type == 'rnaverse-seq')
    
    if training_args.train_from_scratch:
        print('Train from scratch')
        config = RNAverseConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
            problem_type=problem_type,
            is_seq_only_model=is_seq_only,
        )
        if is_seq_only:
            model = RNAverseSeqForSequenceClassificationWithStr(config, fusion_mode=training_args.fusion_mode)
        else:
            model = RNAverseForSequenceClassificationMeanPool(config)
    else:
        print(f'Loading {training_args.model_type} model')
        if is_seq_only:
            model = RNAverseSeqForSequenceClassificationWithStr.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                problem_type=problem_type,
                fusion_mode=training_args.fusion_mode,
            )
        else:
            model = RNAverseForSequenceClassificationMeanPool.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                problem_type=problem_type,
            )

    if training_args.freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith('dense') or \
                name.startswith('classifier') \
                or name.startswith('transfer') \
                or name.startswith('struct_embed') \
                or name.startswith('LayerNorm'):
                print('unfreeze parameter:', name)
                param.requires_grad = True
            else:
                param.requires_grad = False
    print_trainable_parameters(model)

    # define trainer
    training_args.ddp_find_unused_parameters = True
    training_args.save_model = True
    early_stopping = EarlyStoppingCallback(early_stopping_patience=15)
    training_args.label_names = ["labels"]

    lm_para = []
    head_para = []
    for name, param in model.named_parameters():
        if name.startswith('struct_embed') \
                or name.startswith('dense') \
                or name.startswith('classifier') \
                or name.startswith('transfer') \
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
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

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
