# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Score the published structRFM checkpoints on their paired test sets."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from scripts.config import PROJECT_ROOT, registry_rows
from scripts.tasks import SSP_ROOT, TASKS

MODEL_SRC = PROJECT_ROOT.parents[1] / "model_src"
SSP_CODE = MODEL_SRC / "structRFM" / "tasks" / "seqcls_ssp"
STRUCTRFM_SRC = MODEL_SRC / "structRFM" / "src"
OFFICIAL_ROOT = PROJECT_ROOT / "outputs" / "ssp_structrfm" / "official" / "structRFM"
CHECKPOINTS = {
    "bpRNA1m": "structRFM_SSP_bpRNA1m.pt",
    "RNAStrAlign": "structRFM_SSP_RNAStrAlign.pt",
}
SPLITS = {task: spec["test"] for task, spec in TASKS.items()}


def existing_table(task: str) -> Path | None:
    found = sorted((OFFICIAL_ROOT / task).glob("test_metric_*.csv"))
    return found[0] if found else None


def score_published(task: str, device: torch.device, limit: int | None) -> Path:
    for path in (SSP_CODE, STRUCTRFM_SRC):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import param_turner2004
    from collators import SspCollator
    from metrics import SspMetrics
    from RNAdata import BPseqDataset
    from ss_pred import MixedFold, structRFMForSsp
    from structRFM.data.tokenizer import get_mlm_tokenizer
    from structRFM.model.structRFM import get_model_scale, get_structRFM
    from trainers import SspTrainer

    row = next(item for item in registry_rows() if item["name"] == "structRFM")
    checkpoint = row["path"] / CHECKPOINTS[task]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    backbone = row["path"] / "structRFM_checkpoint"
    out_dir = OFFICIAL_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("test_metric_*.csv"):
        old.unlink()

    tokenizer = get_mlm_tokenizer(max_length=514)
    encoder = get_structRFM(
        from_pretrained=str(backbone),
        output_hidden_states=True,
        tokenizer=tokenizer,
        **get_model_scale("base"),
    )
    pretrained_model = structRFMForSsp(encoder)
    head = MixedFold(
        init_param=param_turner2004,
        max_helix_length=30,
        embed_size=64,
        num_filters=(64, 64, 64, 64, 64, 64, 64, 64),
        filter_size=(5, 3, 5, 3, 5, 3, 5, 3),
        pool_size=(1,),
        dilation=0,
        num_lstm_layers=2,
        num_lstm_units=32,
        num_transformer_layers=0,
        num_hidden_units=(32,),
        num_paired_filters=(64, 64, 64, 64, 64, 64, 64, 64),
        paired_filter_size=(5, 3, 5, 3, 5, 3, 5, 3),
        dropout_rate=0.5,
        fc_dropout_rate=0.5,
        num_att=8,
        pair_join="cat",
        no_split_lr=False,
        n_out_paired_layers=3,
        n_out_unpaired_layers=0,
        exclude_diag=True,
        embed_dim=768,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(payload["model"])
    pretrained_model.load_state_dict(payload["pretrained_model"])
    head.to(device)
    pretrained_model.to(device)

    dataset = BPseqDataset(str(SSP_ROOT / SPLITS[task]))
    if limit is not None:
        dataset.file_paths = dataset.file_paths[:limit]
    print(f"dataset {task} test {len(dataset)}", flush=True)
    args = SimpleNamespace(
        batch_size=1,
        dataloader_num_workers=0,
        device=device,
        output_dir=str(out_dir),
        task_name=task,
        train=False,
    )
    trainer = SspTrainer(
        args=args,
        tokenizer=tokenizer,
        model=head,
        pretrained_model=pretrained_model,
        eval_dataset=dataset,
        data_collator=SspCollator(
            max_seq_len=512,
            tokenizer=tokenizer,
            replace_T=True,
            replace_U=False,
            model_name="structRFM",
        ),
        compute_metrics=SspMetrics(metrics=["F1s", "Accuracy", "Precision", "Recall"]),
    )
    trainer.eval(0)
    written = next(out_dir.glob("test_metric_*.csv"))
    table = pd.read_csv(written)
    mean_f1 = float(table["f1s"].mean())
    destination = out_dir / f"test_metric_0_{mean_f1:.4f}.csv"
    if written != destination:
        written.rename(destination)
    print(f"Test\t{task}\tf1s: {mean_f1:.4f}", flush=True)
    print(f"wrote {destination}", flush=True)
    return destination
