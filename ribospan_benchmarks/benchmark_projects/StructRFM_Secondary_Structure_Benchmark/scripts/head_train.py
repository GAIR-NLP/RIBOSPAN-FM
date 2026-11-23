# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

"""Train the structRFM MixedFold head on cached RIBOSPAN states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from scripts.tasks import MAX_NUCLEOTIDES, prepare_example

SSP_CODE = (
    Path(__file__).resolve().parents[3]
    / "model_src"
    / "structRFM"
    / "tasks"
    / "seqcls_ssp"
)


def _import_structrfm():
    code = str(SSP_CODE)
    if code not in sys.path:
        sys.path.insert(0, code)
    import param_turner2004  # noqa: WPS433
    from losses import StructuredLoss  # noqa: WPS433
    from metrics import SspMetrics, compare_bpseq  # noqa: WPS433
    from ss_pred import MixedFold  # noqa: WPS433

    return param_turner2004, StructuredLoss, SspMetrics, compare_bpseq, MixedFold


HEAD_CONFIG = {
    "max_helix_length": 30,
    "embed_size": 64,
    "num_filters": (64,) * 8,
    "filter_size": (5, 3, 5, 3, 5, 3, 5, 3),
    "pool_size": (1,),
    "dilation": 0,
    "num_lstm_layers": 2,
    "num_lstm_units": 32,
    "num_transformer_layers": 0,
    "num_hidden_units": (32,),
    "num_paired_filters": (64,) * 8,
    "paired_filter_size": (5, 3, 5, 3, 5, 3, 5, 3),
    "dropout_rate": 0.5,
    "fc_dropout_rate": 0.5,
    "num_att": 8,
    "pair_join": "cat",
    "no_split_lr": False,
    "n_out_paired_layers": 3,
    "n_out_unpaired_layers": 0,
    "exclude_diag": True,
}


def build_head(embed_dim: int, device: torch.device):
    param_turner2004, StructuredLoss, _metrics, _compare, MixedFold = _import_structrfm()
    model = MixedFold(
        init_param=param_turner2004,
        embed_dim=embed_dim,
        **HEAD_CONFIG,
    )
    model.to(device)
    loss_fn = StructuredLoss(
        loss_pos_paired=0.5,
        loss_neg_paired=0.005,
        loss_pos_unpaired=0.0,
        loss_neg_unpaired=0.0,
    )
    return model, loss_fn


def _embedding_tensor(path: Path, device: torch.device) -> torch.Tensor:
    array = np.load(path).astype(np.float32, copy=False)
    if array.shape[0] > MAX_NUCLEOTIDES + 1:
        raise RuntimeError(f"{path} has {array.shape[0]} positions")
    return torch.from_numpy(array).unsqueeze(0).to(device)


def train_one(
    model,
    loss_fn,
    optimizer,
    seq: str,
    pairs: np.ndarray,
    embedding_path: Path,
    device: torch.device,
) -> float:
    embeddings = _embedding_tensor(embedding_path, device)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model, (seq,), (pairs,), embeddings)
    value = float(loss.detach().cpu())
    if value > 0.0 and np.isfinite(value):
        loss.backward()
        optimizer.step()
    return value


def predict_one(model, seq: str, pairs: np.ndarray, embedding_path: Path, device: torch.device):
    _param, _loss, SspMetrics, compare_bpseq, _fold = _import_structrfm()
    embeddings = _embedding_tensor(embedding_path, device)
    with torch.no_grad():
        _score, _dbn, pred_pairs = model((seq,), embeddings)
    tp, tn, fp, fn = compare_bpseq(pairs, pred_pairs[0])
    # Same metric order as structRFM's SSP default: F1s,Accuracy,Precision,Recall.
    metrics = SspMetrics(["F1s", "Accuracy", "Precision", "Recall"])
    return metrics(tp, tn, fp, fn)


def save_checkpoint(path: Path, model, epoch: int, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "f1": score}, path)


def load_checkpoint(path: Path, model) -> int:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    return int(payload.get("epoch", -1)) + 1


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
