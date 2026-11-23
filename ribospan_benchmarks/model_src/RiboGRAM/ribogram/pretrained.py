# Copyright 2026 Beijing Academy of Artificial Intelligence (BAAI)
# Apache-2.0

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Tuple, Union

import torch
from torch import nn

from .alphabet import (
    Alphabet,
    InteractionAlphabet,
    InteractionBatchConverter,
    InteractionTask,
    RNABatchConverter,
)
from .model import RiboGRAM, RiboGRAMInteraction


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int = 12
    embed_dim: int = 1280
    attention_heads: int = 20
    expanded_vocab_size: int = 38


DEFAULT_MODEL_CONFIG = ModelConfig()


class Architecture(str, Enum):
    RIBOGRAM = "RiboGRAM"
    PLM_CDS = "PLM-CDS"
    PLM_MRNA = "PLM-mRNA"
    RPI = "RiboGRAM-RPI"
    RPIPLUS = "RiboGRAM-RPIplus"
    RRI = "RiboGRAM-RRI"
    RCI = "RiboGRAM-RCI"

    @classmethod
    def parse(cls, value: Union[str, "Architecture"]) -> "Architecture":
        if isinstance(value, cls):
            return value
        normalized = value.casefold().replace("_", "-").replace("^", "")
        aliases = {
            "ribogram": cls.RIBOGRAM,
            "star": cls.RIBOGRAM,
            "plm-cds": cls.PLM_CDS,
            "ribogram-cds": cls.PLM_CDS,
            "plm-mrna": cls.PLM_MRNA,
            "ribogram-mrna": cls.PLM_MRNA,
            "ribogram-rpi": cls.RPI,
            "rpi": cls.RPI,
            "ribogram-rpiplus": cls.RPIPLUS,
            "rpiplus": cls.RPIPLUS,
            "ribogram-rri": cls.RRI,
            "rri": cls.RRI,
            "ribogram-rci": cls.RCI,
            "rci": cls.RCI,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown architecture {value!r}; choose one of: {choices}"
            ) from error

    @property
    def interaction_task(self) -> InteractionTask:
        if self in {Architecture.RPI, Architecture.RPIPLUS}:
            return InteractionTask.RPI
        if self is Architecture.RRI:
            return InteractionTask.RRI
        if self is Architecture.RCI:
            return InteractionTask.RCI
        raise ValueError(f"{self.value} is not an interaction architecture")

    @property
    def is_base_model(self) -> bool:
        return self in {
            Architecture.RIBOGRAM,
            Architecture.PLM_CDS,
            Architecture.PLM_MRNA,
        }


Model = Union[RiboGRAM, RiboGRAMInteraction]
Converter = Union[RNABatchConverter, InteractionBatchConverter]


def build_model_and_alphabet(
    architecture: Union[str, Architecture] = Architecture.RIBOGRAM,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> Tuple[Union[Alphabet, InteractionAlphabet], Converter, Model]:
    """Construct a model and its matching inference converter."""
    architecture = Architecture.parse(architecture)

    if architecture.is_base_model:
        alphabet = Alphabet.from_architecture(architecture.value)
        converter = alphabet.get_batch_converter()
        model: Model = RiboGRAM(
            alphabet_rna=alphabet.model_config(model_config.embed_dim),
            num_layers=model_config.num_layers,
            embed_dim=model_config.embed_dim,
            attention_heads=model_config.attention_heads,
        )
        return alphabet, converter, model

    interaction_alphabet = InteractionAlphabet.from_architecture(
        architecture.value
    )
    interaction_converter = interaction_alphabet.get_batch_converter(
        architecture.interaction_task
    )
    interaction_model = RiboGRAMInteraction(
        alphabet_rna=interaction_alphabet.model_config(model_config.embed_dim),
        num_layers=model_config.num_layers,
        embed_dim=model_config.embed_dim,
        attention_heads=model_config.attention_heads,
        expanded_vocab_size=model_config.expanded_vocab_size,
    )
    return interaction_alphabet, interaction_converter, interaction_model


def load_model_and_alphabet(
    checkpoint_path: Union[str, Path],
    architecture: Union[str, Architecture] = Architecture.RIBOGRAM,
    *,
    strict: bool = True,
    weights_only: bool = False,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> Tuple[Union[Alphabet, InteractionAlphabet], Converter, Model]:
    """Build and strictly load a base or interaction checkpoint on CPU."""
    architecture = Architecture.parse(architecture)
    path = _resolve_checkpoint_path(checkpoint_path, architecture)
    alphabet, converter, model = build_model_and_alphabet(
        architecture=architecture,
        model_config=model_config,
    )
    load_checkpoint(
        model,
        path,
        strict=strict,
        weights_only=weights_only,
    )
    model.eval()
    return alphabet, converter, model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    *,
    strict: bool = True,
    weights_only: bool = False,
) -> nn.Module:
    """Load a direct or wrapped state dictionary into an existing model."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            str(path),
            map_location="cpu",
            weights_only=weights_only,
        )
    except TypeError:
        # Compatibility with older PyTorch releases without ``weights_only``.
        checkpoint = torch.load(str(path), map_location="cpu")

    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return model


def initialize_interaction_from_base(
    model: RiboGRAMInteraction,
    checkpoint_path: Union[str, Path],
    *,
    weights_only: bool = False,
) -> RiboGRAMInteraction:
    """Initialize shared parameters and the first nine expanded rows from RiboGRAM."""
    load_checkpoint(
        model,
        checkpoint_path,
        strict=False,
        weights_only=weights_only,
    )
    with torch.no_grad():
        model.rna_tokens_expend.weight[: model.rna_alphabet_size].copy_(
            model.rna_tokens.weight
        )
    return model


def extract_state_dict(checkpoint: Any) -> "OrderedDict[str, torch.Tensor]":
    """Extract weights and remove one DistributedDataParallel prefix."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")

    cleaned: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("Checkpoint parameter names must be strings")
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def _resolve_checkpoint_path(
    checkpoint_path: Union[str, Path], architecture: Architecture
) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        path = path / f"{architecture.value}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path
