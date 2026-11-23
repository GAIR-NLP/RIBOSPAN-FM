#!/usr/bin/env python3
"""Unified inference examples for the RiboGRAM model family.

The module exposes reusable Python functions and a small command-line interface
for sequence embeddings, attention maps, ncRNA-13/ncRNA-88 classification, and
the RPI/RPIplus/RRI/RCI interaction tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from ribogram import Architecture, load_model_and_alphabet


PathLike = Union[str, Path]
LabeledSequence = Tuple[str, str]
InteractionPair = Tuple[str, str, str]

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINTS: Mapping[Architecture, Path] = {
    Architecture.RIBOGRAM: CHECKPOINT_DIR / "RiboGRAM.pkl",
    Architecture.PLM_CDS: CHECKPOINT_DIR / "PLM-CDS.pkl",
    Architecture.PLM_MRNA: CHECKPOINT_DIR / "PLM-mRNA.pkl",
    Architecture.RPI: CHECKPOINT_DIR / "RiboGRAM-RPI.pkl",
    Architecture.RPIPLUS: CHECKPOINT_DIR / "RiboGRAM-RPI_plus.pkl",
    Architecture.RRI: CHECKPOINT_DIR / "RiboGRAM-RRI.pkl",
    Architecture.RCI: CHECKPOINT_DIR / "RiboGRAM-RCI.pkl",
}
DEFAULT_RIBOGRAM_CHECKPOINT = DEFAULT_CHECKPOINTS[Architecture.RIBOGRAM]
DEFAULT_NCRNA13_HEAD = CHECKPOINT_DIR / "ribogram_ncrna13_head.pt"
DEFAULT_NCRNA88_HEAD = CHECKPOINT_DIR / "ribogram_ncrna88_head.pt"

DEFAULT_EMBEDDING_SEQUENCES: Tuple[LabeledSequence, ...] = (
    (
        "RNA_sequence_1",
        "GGAGCUCAGCCUUCACUGCGUGGCUUCGUUAUGACAUCGAUAUAUCUGCUUCGGCAGGCCUAG"
        "GUGCCUUUACCACGGUCGGAUCCAC",
    ),
    (
        "RNA_sequence_2",
        "GGAGCUCAGCCUUCACUGCGCAGAGGUCCGCUUGAAAACGUCCUGCUGCUUCGGCAGACCGUG"
        "GUCGUGCACUGCCGGCCGGAUCCAC",
    ),
)


@dataclass(frozen=True)
class AttentionCase:
    """Named RNA used by the manuscript attention examples."""

    name: str
    accession: str
    sequence: str


ATTENTION_CASES: Mapping[str, AttentionCase] = {
    "cd-box": AttentionCase(
        name="CD-box_snoZ39",
        accession="RF00341; AJ240057.1/1-60",
        sequence=(
            "GUACAUGUGAUGAAGCAAAUCAGUAUGAAUAAAAUGAUACUGUAAACGCUUUCUGAUGUA"
        ),
    ),
    "trna": AttentionCase(
        name="tRNA",
        accession="X14835.1/6927-7002",
        sequence=(
            "GCCGGGGUCGCCUAGCCUGGUCAAGGGCGCCGGACUCAUAAUCCGGUCUUCCCGGGUUCGAA"
            "UCCCGGCCCCGGCA"
        ),
    ),
    "haca-box": AttentionCase(
        name="HACA-box_snoF1_F2",
        accession="RF00482; X67711.2/1771-1910",
        sequence=(
            "UUAUGCCAUUUGGUAUCUGACUCAUUGUCUACUUUUGUGACUAUGUGUCUUGAAAUAUGGCA"
            "UAUAAUAAGGGGGAACGGAUCUUGCGACAUUGAUGGCUCCUGUAUGAGCCUCCAAUUACUAUA"
            "UAGCCGUUUACAUGU"
        ),
    ),
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _normalize_rna(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper().replace("T", "U")
    if not normalized:
        raise ValueError("RNA sequences must not be empty")
    return normalized


def _normalize_protein(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Protein sequences must not be empty")
    return normalized


def _normalize_dna(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper().replace("U", "T")
    if not normalized:
        raise ValueError("DNA sequences must not be empty")
    return normalized


def default_checkpoint(
    architecture: Union[str, Architecture],
) -> Path:
    """Return the release checkpoint bundled for an architecture."""
    return DEFAULT_CHECKPOINTS[Architecture.parse(architecture)]


def _biological_mask(tokens: Tensor, alphabet: Any) -> Tensor:
    return (
        tokens.ne(alphabet.padding_idx)
        & tokens.ne(alphabet.cls_idx)
        & tokens.ne(alphabet.eos_idx)
    )


def inference_sequence_embeddings(
    checkpoint_path: PathLike,
    sequences: Sequence[LabeledSequence] = DEFAULT_EMBEDDING_SEQUENCES,
    *,
    architecture: Union[str, Architecture] = Architecture.RIBOGRAM,
    layer: int = 12,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Extract one token embedding matrix and one mean vector per RNA.

    Special tokens and padding are removed from ``token_embeddings``. The
    returned tensors are detached CPU tensors and can be saved directly with
    :func:`torch.save`.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    records = [(label, _normalize_rna(sequence)) for label, sequence in sequences]
    architecture = Architecture.parse(architecture)
    if not architecture.is_base_model:
        raise ValueError("Embedding extraction requires RiboGRAM, PLM-CDS, or PLM-mRNA")
    run_device = _resolve_device(device)
    alphabet, converter, model = load_model_and_alphabet(
        checkpoint_path,
        architecture=architecture,
    )
    if not 0 <= layer <= model.num_layers:
        raise ValueError(f"layer must be between 0 and {model.num_layers}")
    model = model.to(run_device)

    results: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            labels, normalized_sequences, tokens = converter(
                records[start : start + batch_size]
            )
            tokens = tokens.to(run_device)
            output = model(tokens, repr_layers=[layer])
            representations = output["hidden_representations"][layer]
            mask = _biological_mask(tokens, alphabet)
            for index, label in enumerate(labels):
                token_embeddings = representations[index][mask[index]].float().cpu()
                results.append(
                    {
                        "label": str(label),
                        "sequence": normalized_sequences[index],
                        "layer": layer,
                        "token_embeddings": token_embeddings,
                        "mean_embedding": token_embeddings.mean(dim=0),
                    }
                )
    return results


def average_product_correction(
    attention: Tensor,
    *,
    positive_only: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Apply average-product correction to a square attention matrix."""
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError("attention must be a square two-dimensional tensor")
    row_mean = attention.mean(dim=-1, keepdim=True)
    column_mean = attention.mean(dim=-2, keepdim=True)
    global_mean = attention.mean().clamp_min(eps)
    corrected = attention - (row_mean * column_mean) / global_mean
    return corrected.clamp_min(0.0) if positive_only else corrected


def inference_attention(
    checkpoint_path: PathLike,
    sequence: str,
    *,
    architecture: Union[str, Architecture] = Architecture.RIBOGRAM,
    layer: int = 12,
    head: Optional[int] = None,
    device: str = "auto",
    symmetrize: bool = False,
    positive_apc: bool = True,
) -> Dict[str, Any]:
    """Calculate per-head and APC-corrected RNA attention.

    The defaults reproduce the historical manuscript extraction convention:
    remove ``<cls>/<eos>/<pad>``, select the final head of the requested layer,
    do not symmetrize, and retain only positive APC values. ``head`` is a
    one-based head number; ``None`` selects the final head.
    """
    return inference_attention_batch(
        checkpoint_path,
        [("attention", sequence)],
        architecture=architecture,
        layer=layer,
        head=head,
        device=device,
        symmetrize=symmetrize,
        positive_apc=positive_apc,
    )[0]


def inference_attention_batch(
    checkpoint_path: PathLike,
    sequences: Sequence[LabeledSequence],
    *,
    architecture: Union[str, Architecture] = Architecture.RIBOGRAM,
    layer: int = 12,
    head: Optional[int] = None,
    device: str = "auto",
    symmetrize: bool = False,
    positive_apc: bool = True,
) -> List[Dict[str, Any]]:
    """Calculate one layer/head attention map for multiple RNA sequences.

    All heads from the requested layer are retained in the returned tensor
    bundle. APC is computed from the selected head, which defaults to the final
    head in accordance with the manuscript analysis scripts.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")

    records = []
    for label, sequence in sequences:
        normalized = _normalize_rna(sequence)
        invalid = sorted(set(normalized).difference("ACGU"))
        if invalid:
            raise ValueError(
                f"Attention input {label!r} contains unsupported symbols: {invalid}"
            )
        records.append((label, normalized))

    architecture = Architecture.parse(architecture)
    if not architecture.is_base_model:
        raise ValueError("Attention requires RiboGRAM, PLM-CDS, or PLM-mRNA")
    run_device = _resolve_device(device)
    alphabet, converter, model = load_model_and_alphabet(
        checkpoint_path,
        architecture=architecture,
    )
    if not 1 <= layer <= model.num_layers:
        raise ValueError(f"layer must be between 1 and {model.num_layers}")
    model = model.to(run_device)

    labels, normalized_sequences, tokens = converter(records)
    tokens = tokens.to(run_device)
    with torch.inference_mode():
        output = model(tokens, need_head_weights=True)

    all_attention = output["attentions"]
    biological_mask = _biological_mask(tokens, alphabet)
    results = []
    for index, label in enumerate(labels):
        keep = torch.where(biological_mask[index])[0]
        per_head = all_attention[index, layer - 1].float()
        per_head = per_head.index_select(-2, keep).index_select(-1, keep)
        per_head = torch.nan_to_num(per_head, nan=0.0)
        num_heads = int(per_head.shape[0])
        selected_head = num_heads if head is None else int(head)
        if not 1 <= selected_head <= num_heads:
            raise ValueError(f"head must be between 1 and {num_heads}")

        head_attention = per_head[selected_head - 1]
        mean_attention = per_head.mean(dim=0)
        if symmetrize:
            head_attention = 0.5 * (
                head_attention + head_attention.transpose(-1, -2)
            )
            mean_attention = 0.5 * (
                mean_attention + mean_attention.transpose(-1, -2)
            )
        apc_attention = average_product_correction(
            head_attention,
            positive_only=positive_apc,
        )
        results.append(
            {
                "label": str(label),
                "sequence": normalized_sequences[index],
                "layer": layer,
                "head": selected_head,
                "num_heads": num_heads,
                "symmetrized": symmetrize,
                "positive_apc": positive_apc,
                "per_head_attention": per_head.cpu(),
                "head_attention": head_attention.cpu(),
                "mean_attention": mean_attention.cpu(),
                "apc_attention": apc_attention.cpu(),
            }
        )
    return results


def save_attention_outputs(
    result: Mapping[str, Any],
    output_prefix: PathLike,
    *,
    plot: bool = False,
) -> Dict[str, Path]:
    """Save a complete tensor bundle, APC matrix CSV, metadata, and heatmap."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    tensor_path = prefix.with_suffix(".pt")
    csv_path = prefix.with_suffix(".csv")
    metadata_path = prefix.with_suffix(".json")
    torch.save(dict(result), tensor_path)

    matrix = result["apc_attention"]
    if not isinstance(matrix, Tensor):
        raise TypeError("result['apc_attention'] must be a torch.Tensor")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.tolist())

    metadata = {
        "sequence": result["sequence"],
        "length": len(result["sequence"]),
        "layer": result["layer"],
        "head": result["head"],
        "num_heads": result["num_heads"],
        "attention_source": "selected_head",
        "symmetrized": result["symmetrized"],
        "positive_apc": result["positive_apc"],
        "per_head_shape": list(result["per_head_attention"].shape),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    paths = {
        "tensors": tensor_path,
        "matrix": csv_path,
        "metadata": metadata_path,
    }
    if plot:
        paths["heatmap"] = _save_attention_heatmap(
            matrix,
            prefix.with_suffix(".png"),
            layer=int(result["layer"]),
            head=int(result["head"]),
            positive_only=bool(result["positive_apc"]),
        )
    return paths


def _one_based_position_ticks(
    length: int,
    *,
    max_ticks: int = 8,
) -> Tuple[List[int], List[int]]:
    """Return zero-based image coordinates with one-based sequence labels."""
    if length < 1:
        raise ValueError("length must be positive")
    raw_step = max(1.0, length / max_ticks)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(
        int(multiplier * magnitude)
        for multiplier in (1, 2, 5, 10)
        if multiplier * magnitude >= raw_step
    )
    labels = [1, *range(step, length + 1, step)]
    if labels[-1] != length:
        labels.append(length)
    labels = list(dict.fromkeys(labels))
    return [label - 1 for label in labels], labels


def _save_attention_heatmap(
    matrix: Tensor,
    path: Path,
    *,
    layer: int,
    head: int,
    positive_only: bool,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib; install it or omit --plot"
        ) from error

    length = int(matrix.shape[0])
    tick_positions, tick_labels = _one_based_position_ticks(length)
    figure, axis = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    image = axis.imshow(
        matrix.numpy(),
        cmap="magma",
        interpolation="nearest",
        origin="upper",
        vmin=0.0 if positive_only else None,
    )
    axis.set_xticks(tick_positions, tick_labels)
    axis.set_yticks(tick_positions, tick_labels)
    axis.set_xlim(-0.5, length - 0.5)
    axis.set_ylim(length - 0.5, -0.5)
    axis.set_xlabel("Nucleotide position (5′ → 3′)")
    axis.set_ylabel("Nucleotide position (5′ → 3′)")
    axis.set_title(f"Layer {layer}, Head {head} attention (APC)")
    figure.colorbar(
        image,
        ax=axis,
        label=f"Layer {layer}, Head {head} attention",
    )
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


class NcRNAClassificationHead(nn.Module):
    """MLP architecture shared by the released ncRNA classifier heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, num_classes)
            return

        layers: List[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend(
                (
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


def pooled_ncrna_features(
    representation: Tensor,
    biological_mask: Tensor,
) -> Dict[str, Tensor]:
    """Reproduce the pooling used to train the ncRNA-13 and ncRNA-88 heads."""
    if representation.ndim != 3 or biological_mask.ndim != 2:
        raise ValueError("Expected representation [B,L,C] and mask [B,L]")
    if representation.shape[:2] != biological_mask.shape:
        raise ValueError("Representation and mask dimensions do not match")
    if not biological_mask.any(dim=1).all():
        raise ValueError("Every sequence must contain at least one nucleotide")

    features = representation.float()
    mask = biological_mask.bool()
    expanded_mask = mask.unsqueeze(-1)
    lengths = mask.sum(dim=1)
    denominator = lengths.unsqueeze(-1).float()
    mean = (features * expanded_mask).sum(dim=1) / denominator
    second_moment = (features.square() * expanded_mask).sum(dim=1) / denominator

    pooled = {
        "mean": mean,
        "std": (second_moment - mean.square()).clamp_min(0.0).sqrt(),
        "max": features.masked_fill(~expanded_mask, -torch.inf).amax(dim=1),
        "cls": features[:, 0],
        "eos": features[
            torch.arange(features.shape[0], device=features.device),
            lengths + 1,
        ],
        "first": features[:, 1],
        "last": features[
            torch.arange(features.shape[0], device=features.device),
            lengths,
        ],
    }

    positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
    biological_positions = (positions - 1).clamp_min(0)
    for quarter in range(4):
        lower = (lengths * quarter) // 4
        upper = (lengths * (quarter + 1)) // 4
        quarter_mask = (
            mask
            & (biological_positions >= lower.unsqueeze(1))
            & (biological_positions < upper.unsqueeze(1))
        )
        pooled[f"q{quarter + 1}"] = (
            features * quarter_mask.unsqueeze(-1)
        ).sum(dim=1) / quarter_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()

    pooled["length"] = lengths.float().log1p().unsqueeze(1)
    return pooled


def inference_ncrna_classification(
    backbone_checkpoint: PathLike,
    head_checkpoint: PathLike,
    sequences: Sequence[LabeledSequence],
    *,
    expected_num_classes: Optional[int] = None,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Classify ncRNAs with a released head over frozen Layer-12 features."""
    if not sequences:
        raise ValueError("sequences must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    records = [(label, _normalize_rna(sequence)) for label, sequence in sequences]
    run_device = _resolve_device(device)
    bundle = _torch_load(head_checkpoint)
    if not isinstance(bundle, Mapping):
        raise TypeError("The ncRNA classifier checkpoint must contain a mapping")
    if "class_names" not in bundle or "members" not in bundle:
        raise ValueError("Classifier checkpoint lacks class_names or members")

    class_names = list(bundle["class_names"])
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Classifier class_names must be non-empty and unique")
    if expected_num_classes is not None and len(class_names) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(class_names)}"
        )

    artifacts = list(bundle["members"])
    if not artifacts:
        raise ValueError("Classifier checkpoint contains no head members")

    members = []
    for artifact in artifacts:
        head_config = artifact["head"]
        num_classes = int(artifact["num_classes"])
        if num_classes != len(class_names):
            raise ValueError(
                "Head output size does not match the classifier class mapping"
            )
        head = NcRNAClassificationHead(
            input_dim=int(artifact["input_dim"]),
            hidden_dim=int(head_config["hidden"]),
            depth=int(head_config["depth"]),
            dropout=float(head_config["dropout"]),
            num_classes=num_classes,
        )
        head.load_state_dict(artifact["model_state_dict"], strict=True)
        head = head.to(run_device).eval()
        members.append(
            (
                head,
                tuple(artifact["feature_names"]),
                artifact["center"].to(run_device),
                artifact["scale"].to(run_device),
            )
        )

    alphabet, converter, backbone = load_model_and_alphabet(
        backbone_checkpoint,
        architecture=Architecture.RIBOGRAM,
    )
    backbone = backbone.to(run_device)

    predictions: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            labels, normalized_sequences, tokens = converter(batch)
            tokens = tokens.to(run_device)
            biological_mask = _biological_mask(tokens, alphabet)
            with _autocast(run_device):
                representation = backbone(tokens)["representation"]
            feature_map = pooled_ncrna_features(
                representation,
                biological_mask,
            )

            member_probabilities = []
            for head, feature_names, center, scale in members:
                features = torch.cat(
                    [feature_map[name] for name in feature_names],
                    dim=1,
                )
                standardized = (features - center) / scale.clamp_min(1e-5)
                member_probabilities.append(head(standardized).softmax(dim=-1))
            probabilities = torch.stack(member_probabilities).mean(dim=0).cpu()

            for index, label in enumerate(labels):
                predicted_index = int(probabilities[index].argmax().item())
                predictions.append(
                    {
                        "id": str(label),
                        "sequence": normalized_sequences[index],
                        "num_classes": len(class_names),
                        "predicted_label_id": predicted_index,
                        "predicted_label": class_names[predicted_index],
                        "confidence": float(probabilities[index, predicted_index]),
                        "probabilities": {
                            name: float(probabilities[index, class_index])
                            for class_index, name in enumerate(class_names)
                        },
                    }
                )
    return predictions


def inference_ncrna13(
    backbone_checkpoint: PathLike,
    head_checkpoint: PathLike,
    sequences: Sequence[LabeledSequence],
    *,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Predict the 13 broad ncRNA categories."""
    return inference_ncrna_classification(
        backbone_checkpoint,
        head_checkpoint,
        sequences,
        expected_num_classes=13,
        batch_size=batch_size,
        device=device,
    )


def inference_ncrna88(
    backbone_checkpoint: PathLike,
    head_checkpoint: PathLike,
    sequences: Sequence[LabeledSequence],
    *,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Predict the 88 Rfam-family classes."""
    return inference_ncrna_classification(
        backbone_checkpoint,
        head_checkpoint,
        sequences,
        expected_num_classes=88,
        batch_size=batch_size,
        device=device,
    )


def inference_interaction(
    checkpoint_path: PathLike,
    architecture: Union[str, Architecture],
    pairs: Sequence[InteractionPair],
    *,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Predict RPI/RPIplus, RRI, or RCI interaction probabilities."""
    architecture = Architecture.parse(architecture)
    if architecture.is_base_model:
        raise ValueError("Use an RPI/RPIplus, RRI, or RCI architecture")
    if not pairs:
        raise ValueError("pairs must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    normalized_records: List[InteractionPair] = []
    for identifier, rna, partner in pairs:
        normalized_rna = _normalize_rna(rna)
        if architecture in {Architecture.RPI, Architecture.RPIPLUS}:
            normalized_partner = _normalize_protein(partner)
        elif architecture is Architecture.RRI:
            normalized_partner = _normalize_rna(partner)
        else:
            normalized_partner = _normalize_dna(partner)
        normalized_records.append(
            (str(identifier), normalized_rna, normalized_partner)
        )

    run_device = _resolve_device(device)
    _, converter, model = load_model_and_alphabet(
        checkpoint_path,
        architecture=architecture,
    )
    model = model.to(run_device)

    predictions: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(normalized_records), batch_size):
            labels, normalized_pairs, tokens = converter(
                normalized_records[start : start + batch_size]
            )
            logits = model(tokens.to(run_device))["pair_logits"].squeeze(-1)
            logits = logits.float().cpu()
            probabilities = logits.sigmoid()

            for index, label in enumerate(labels):
                probability = float(probabilities[index])
                rna, partner = normalized_pairs[index]
                predictions.append(
                    {
                        "id": str(label),
                        "architecture": architecture.value,
                        "rna": rna,
                        "partner": partner,
                        "logit": float(logits[index]),
                        "probability": probability,
                        "prediction": int(probability >= threshold),
                        "threshold": threshold,
                    }
                )
    return predictions


def inference_rpi(
    checkpoint_path: PathLike,
    pairs: Sequence[InteractionPair],
    *,
    rpiplus: bool = False,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Predict RNA-protein interactions with RPI or RPIplus weights."""
    architecture = Architecture.RPIPLUS if rpiplus else Architecture.RPI
    return inference_interaction(
        checkpoint_path,
        architecture,
        pairs,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    )


def inference_rri(
    checkpoint_path: PathLike,
    pairs: Sequence[InteractionPair],
    *,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Predict RNA-RNA interactions."""
    return inference_interaction(
        checkpoint_path,
        Architecture.RRI,
        pairs,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    )


def inference_rci(
    checkpoint_path: PathLike,
    pairs: Sequence[InteractionPair],
    *,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """Predict RNA-chromatin interactions from RNA and forward DNA."""
    return inference_interaction(
        checkpoint_path,
        Architecture.RCI,
        pairs,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    )


def _torch_load(path: PathLike) -> Any:
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        return torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint), map_location="cpu")


def _parse_labeled_sequences(values: Optional[Sequence[str]]) -> List[LabeledSequence]:
    if not values:
        return list(DEFAULT_EMBEDDING_SEQUENCES)
    records = []
    for index, value in enumerate(values, start=1):
        if "=" in value:
            label, sequence = value.split("=", 1)
        else:
            label, sequence = f"sequence_{index}", value
        records.append((label, sequence))
    return records


def _safe_stem(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_") or "attention"


def _write_json(data: Any, output: Optional[PathLike]) -> None:
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if output is None:
        print(serialized, end="")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    print(f"Saved: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    embedding = commands.add_parser("embedding", help="Extract RNA embeddings")
    embedding.add_argument(
        "--checkpoint",
        help="Checkpoint path; defaults to checkpoints/<architecture>.pkl",
    )
    embedding.add_argument(
        "--architecture",
        default=Architecture.RIBOGRAM.value,
        choices=[
            Architecture.RIBOGRAM.value,
            Architecture.PLM_CDS.value,
            Architecture.PLM_MRNA.value,
        ],
    )
    embedding.add_argument(
        "--sequence",
        action="append",
        metavar="ID=RNA",
        help="May be repeated; built-in examples are used when omitted",
    )
    embedding.add_argument("--layer", type=int, default=12)
    embedding.add_argument("--batch-size", type=int, default=8)
    embedding.add_argument("--device", default="auto")
    embedding.add_argument("--output", default="sequence_embeddings.pt")

    attention = commands.add_parser("attention", help="Calculate RNA attention")
    attention.add_argument(
        "--checkpoint",
        help="Checkpoint path; defaults to checkpoints/<architecture>.pkl",
    )
    attention.add_argument(
        "--architecture",
        default=Architecture.RIBOGRAM.value,
        choices=[
            Architecture.RIBOGRAM.value,
            Architecture.PLM_CDS.value,
            Architecture.PLM_MRNA.value,
        ],
    )
    attention_input = attention.add_mutually_exclusive_group()
    attention_input.add_argument(
        "--case",
        choices=[*ATTENTION_CASES, "all"],
        help="Built-in manuscript case; defaults to all cases",
    )
    attention_input.add_argument("--sequence", help="Custom RNA sequence")
    attention.add_argument("--name", default="custom")
    attention.add_argument("--layer", type=int, default=12)
    attention.add_argument(
        "--head",
        type=int,
        help="One-based attention-head number; defaults to the final head",
    )
    attention.add_argument("--device", default="auto")
    attention.add_argument("--symmetrize", action="store_true")
    attention.add_argument("--keep-negative-apc", action="store_true")
    attention.add_argument("--plot", action="store_true")
    attention.add_argument("--output-dir", default="attention_outputs")

    ncrna13 = commands.add_parser("ncrna13", help="Classify 13 ncRNA categories")
    ncrna13.add_argument(
        "--backbone",
        default=str(DEFAULT_RIBOGRAM_CHECKPOINT),
    )
    ncrna13.add_argument("--head", default=str(DEFAULT_NCRNA13_HEAD))
    ncrna13.add_argument(
        "--sequence",
        action="append",
        required=True,
        metavar="ID=RNA",
    )
    ncrna13.add_argument("--batch-size", type=int, default=8)
    ncrna13.add_argument("--device", default="auto")
    ncrna13.add_argument("--output")

    ncrna88 = commands.add_parser("ncrna88", help="Classify 88 Rfam families")
    ncrna88.add_argument(
        "--backbone",
        default=str(DEFAULT_RIBOGRAM_CHECKPOINT),
    )
    ncrna88.add_argument("--head", default=str(DEFAULT_NCRNA88_HEAD))
    ncrna88.add_argument(
        "--sequence",
        action="append",
        required=True,
        metavar="ID=RNA",
    )
    ncrna88.add_argument("--batch-size", type=int, default=8)
    ncrna88.add_argument("--device", default="auto")
    ncrna88.add_argument("--output")

    interaction = commands.add_parser(
        "interaction",
        help="Predict RPI/RPIplus/RRI/RCI interactions",
    )
    interaction.add_argument(
        "--checkpoint",
        help="Checkpoint path; defaults to the selected bundled architecture",
    )
    interaction.add_argument(
        "--architecture",
        required=True,
        choices=[
            Architecture.RPI.value,
            Architecture.RPIPLUS.value,
            Architecture.RRI.value,
            Architecture.RCI.value,
        ],
    )
    interaction.add_argument("--id", default="pair-1")
    interaction.add_argument("--rna", required=True)
    interaction.add_argument(
        "--partner",
        required=True,
        help="Protein for RPI, second RNA for RRI, or forward DNA for RCI",
    )
    interaction.add_argument("--threshold", type=float, default=0.5)
    interaction.add_argument("--batch-size", type=int, default=8)
    interaction.add_argument("--device", default="auto")
    interaction.add_argument("--output")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "embedding":
        records = _parse_labeled_sequences(args.sequence)
        checkpoint = args.checkpoint or default_checkpoint(args.architecture)
        results = inference_sequence_embeddings(
            checkpoint,
            records,
            architecture=args.architecture,
            layer=args.layer,
            batch_size=args.batch_size,
            device=args.device,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, output)
        for result in results:
            print(
                f"{result['label']}: "
                f"token_embeddings={tuple(result['token_embeddings'].shape)}, "
                f"mean_embedding={tuple(result['mean_embedding'].shape)}"
            )
        print(f"Saved: {output}")
        return

    if args.command == "attention":
        if args.sequence is not None:
            cases = [AttentionCase(args.name, "custom", args.sequence)]
        else:
            selected = args.case or "all"
            case_names = list(ATTENTION_CASES) if selected == "all" else [selected]
            cases = [ATTENTION_CASES[name] for name in case_names]

        checkpoint = args.checkpoint or default_checkpoint(args.architecture)
        results = inference_attention_batch(
            checkpoint,
            [(case.name, case.sequence) for case in cases],
            architecture=args.architecture,
            layer=args.layer,
            head=args.head,
            device=args.device,
            symmetrize=args.symmetrize,
            positive_apc=not args.keep_negative_apc,
        )
        for case, result in zip(cases, results):
            result["name"] = case.name
            result["accession"] = case.accession
            prefix = Path(args.output_dir) / _safe_stem(case.name)
            paths = save_attention_outputs(result, prefix, plot=args.plot)
            print(case.name, {name: str(path) for name, path in paths.items()})
        return

    if args.command in {"ncrna13", "ncrna88"}:
        inference_function = (
            inference_ncrna13
            if args.command == "ncrna13"
            else inference_ncrna88
        )
        predictions = inference_function(
            args.backbone,
            args.head,
            _parse_labeled_sequences(args.sequence),
            batch_size=args.batch_size,
            device=args.device,
        )
        _write_json(predictions, args.output)
        return

    checkpoint = args.checkpoint or default_checkpoint(args.architecture)
    predictions = inference_interaction(
        checkpoint,
        args.architecture,
        [(args.id, args.rna, args.partner)],
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )
    _write_json(predictions, args.output)


if __name__ == "__main__":
    main()
