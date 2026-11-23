# Copyright 2026 Beijing Academy of Artificial Intelligence (BAAI)
# Apache-2.0

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


BASE_TOKENS: Tuple[str, ...] = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "A",
    "C",
    "G",
    "U",
    "<mask>",
)

INTERACTION_TOKENS: Tuple[str, ...] = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "R_A",
    "R_C",
    "R_G",
    "R_U",
    "<mask>",
    "P_L",
    "P_A",
    "P_G",
    "P_V",
    "P_S",
    "P_E",
    "P_R",
    "P_T",
    "P_I",
    "P_D",
    "P_P",
    "P_K",
    "P_Q",
    "P_N",
    "P_F",
    "P_Y",
    "P_M",
    "P_H",
    "P_W",
    "P_C",
    "P_X",
    "P_B",
    "P_U",
    "P_Z",
    "P_O",
    "D_A",
    "D_C",
    "D_G",
    "D_T",
)


class InteractionTask(str, Enum):
    RPI = "rpi"
    RRI = "rri"
    RCI = "rci"

    @classmethod
    def parse(cls, value: Union[str, "InteractionTask"]) -> "InteractionTask":
        if isinstance(value, cls):
            return value
        normalized = value.casefold().replace("_", "-").replace("^", "")
        prefix = "ribogram-"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
        if normalized == "rpiplus":
            normalized = "rpi"
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(task.value for task in cls)
            raise ValueError(
                f"Unknown interaction task {value!r}; choose one of: {choices}"
            ) from error


class _Vocabulary:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.all_toks = list(tokens)
        self.tok_to_idx = {
            token: index for index, token in enumerate(self.all_toks)
        }
        self.cls_idx = self.tok_to_idx["<cls>"]
        self.padding_idx = self.tok_to_idx["<pad>"]
        self.eos_idx = self.tok_to_idx["<eos>"]
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.mask_idx = self.tok_to_idx["<mask>"]

    def __len__(self) -> int:
        return len(self.all_toks)

    def get_idx(self, token: str) -> int:
        return self.tok_to_idx.get(token, self.unk_idx)

    def get_tok(self, index: int) -> str:
        return self.all_toks[index]

    def to_dict(self) -> Dict[str, int]:
        return self.tok_to_idx.copy()

    def decode(self, token_ids: Iterable[int]) -> List[str]:
        return [self.get_tok(index) for index in token_ids]

    def model_config(self, embed_dim: int = 1280) -> Dict[str, int]:
        return {
            "rna_alphabet_size": 9,
            "rna_padding_idx": self.padding_idx,
            "rna_embedding_size": embed_dim,
        }


class Alphabet(_Vocabulary):
    """Nine-token vocabulary used by the base RiboGRAM checkpoint."""

    def __init__(self) -> None:
        super().__init__(BASE_TOKENS)
        self.prepend_bos = True
        self.append_eos = True

    @classmethod
    def from_architecture(cls, name: str = "RiboGRAM") -> "Alphabet":
        normalized = name.casefold().replace("_", "-")
        if normalized not in {"ribogram", "star", "plm-cds", "plm-mrna"}:
            raise ValueError(f"Unknown base architecture: {name}")
        return cls()

    @staticmethod
    def normalize_rna(sequence: str) -> str:
        return "".join(sequence.split()).upper().replace("T", "U")

    def tokenize(self, sequence: str) -> List[str]:
        text = "".join(sequence.split())
        tokens: List[str] = []
        position = 0
        while position < len(text):
            if text[position] == "<":
                end = text.find(">", position + 1)
                if end != -1:
                    special = text[position : end + 1].lower()
                    if special in self.tok_to_idx:
                        tokens.append(special)
                        position = end + 1
                        continue
            nucleotide = text[position].upper()
            tokens.append("U" if nucleotide == "T" else nucleotide)
            position += 1
        return tokens

    def encode(self, sequence: str) -> List[int]:
        return [self.get_idx(token) for token in self.tokenize(sequence)]

    def get_batch_converter(
        self, truncation_seq_length: Optional[int] = None
    ) -> "RNABatchConverter":
        return RNABatchConverter(self, truncation_seq_length)


class InteractionAlphabet(_Vocabulary):
    """Shared 38-token vocabulary for RPI/RPIplus, RRI and RCI."""

    def __init__(self) -> None:
        super().__init__(INTERACTION_TOKENS)

    @classmethod
    def from_architecture(cls, name: str) -> "InteractionAlphabet":
        InteractionTask.parse(name)
        return cls()

    @staticmethod
    def normalize_rna(sequence: str) -> str:
        return "".join(sequence.split()).upper().replace("T", "U")

    @staticmethod
    def normalize_protein(sequence: str) -> str:
        return "".join(sequence.split()).upper()

    @staticmethod
    def normalize_dna(sequence: str) -> str:
        return "".join(sequence.split()).upper().replace("U", "T")

    def encode_rna(self, sequence: str) -> List[int]:
        return [
            self.get_idx(f"R_{base}")
            for base in self.normalize_rna(sequence)
        ]

    def encode_protein(self, sequence: str) -> List[int]:
        return [
            self.get_idx(f"P_{residue}")
            for residue in self.normalize_protein(sequence)
        ]

    def encode_dna(self, sequence: str) -> List[int]:
        return [
            self.get_idx(f"D_{base}")
            for base in self.normalize_dna(sequence)
        ]

    def get_batch_converter(
        self,
        task: Union[str, InteractionTask],
        max_rna_length: Optional[int] = None,
        max_partner_length: Optional[int] = None,
    ) -> "InteractionBatchConverter":
        return InteractionBatchConverter(
            alphabet=self,
            task=InteractionTask.parse(task),
            max_rna_length=max_rna_length,
            max_partner_length=max_partner_length,
        )


class RNABatchConverter:
    """Convert ``(label, RNA)`` records into base-model tokens."""

    def __init__(
        self,
        alphabet: Alphabet,
        truncation_seq_length: Optional[int] = None,
    ) -> None:
        _validate_optional_length("truncation_seq_length", truncation_seq_length)
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(
        self, raw_batch: Iterable[Tuple[Any, str]]
    ) -> Tuple[List[Any], List[str], Tensor]:
        batch = list(raw_batch)
        if not batch:
            raise ValueError("raw_batch must not be empty")

        labels, sequences = zip(*batch)
        encoded = [self.alphabet.encode(sequence) for sequence in sequences]
        if self.truncation_seq_length is not None:
            encoded = [
                tokens[: self.truncation_seq_length] for tokens in encoded
            ]

        rows = [
            [self.alphabet.cls_idx, *tokens, self.alphabet.eos_idx]
            for tokens in encoded
        ]
        return list(labels), list(sequences), _pad_rows(
            rows, self.alphabet.padding_idx
        )


class InteractionBatchConverter:
    """Encode the task-specific concatenated inputs used during post-training."""

    def __init__(
        self,
        alphabet: InteractionAlphabet,
        task: InteractionTask,
        max_rna_length: Optional[int] = None,
        max_partner_length: Optional[int] = None,
    ) -> None:
        _validate_optional_length("max_rna_length", max_rna_length)
        _validate_optional_length("max_partner_length", max_partner_length)
        self.alphabet = alphabet
        self.task = task
        self.max_rna_length = max_rna_length
        self.max_partner_length = max_partner_length

    def __call__(
        self, raw_batch: Iterable[Tuple[Any, str, str]]
    ) -> Tuple[List[Any], List[Tuple[str, str]], Tensor]:
        batch = list(raw_batch)
        if not batch:
            raise ValueError("raw_batch must not be empty")

        labels: List[Any] = []
        pairs: List[Tuple[str, str]] = []
        rows: List[List[int]] = []
        for label, rna, partner in batch:
            labels.append(label)
            pairs.append((rna, partner))
            rows.append(self._encode_pair(rna, partner))

        return labels, pairs, _pad_rows(rows, self.alphabet.padding_idx)

    def _encode_pair(self, rna: str, partner: str) -> List[int]:
        rna_ids = self.alphabet.encode_rna(rna)
        if self.max_rna_length is not None:
            rna_ids = rna_ids[: self.max_rna_length]

        cls_id = self.alphabet.cls_idx
        eos_id = self.alphabet.eos_idx

        if self.task is InteractionTask.RPI:
            partner_ids = self.alphabet.encode_protein(partner)
            if self.max_partner_length is not None:
                partner_ids = partner_ids[: self.max_partner_length]
            return [cls_id, *rna_ids, eos_id, *partner_ids, eos_id]

        if self.task is InteractionTask.RRI:
            partner_ids = self.alphabet.encode_rna(partner)
            if self.max_partner_length is not None:
                partner_ids = partner_ids[: self.max_partner_length]
            return [cls_id, *rna_ids, eos_id, *partner_ids, eos_id]

        dna = self.alphabet.normalize_dna(partner)
        if self.max_partner_length is not None:
            dna = dna[: self.max_partner_length]
        dna_forward = self.alphabet.encode_dna(dna)
        dna_reverse = self.alphabet.encode_dna(reverse_complement(dna))
        return [
            cls_id,
            *rna_ids,
            eos_id,
            *dna_forward,
            eos_id,
            *dna_reverse,
            eos_id,
        ]


def reverse_complement(sequence: str) -> str:
    """Return the DNA reverse complement, preserving common IUPAC symbols."""
    normalized = InteractionAlphabet.normalize_dna(sequence)
    table = str.maketrans("ACGTRYMKBDHVN", "TGCAYRKMVHDBN")
    return normalized.translate(table)[::-1]


def _validate_optional_length(name: str, value: Optional[int]) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive")


def _pad_rows(rows: Sequence[Sequence[int]], padding_idx: int) -> Tensor:
    maximum_length = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), maximum_length),
        padding_idx,
        dtype=torch.long,
    )
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return tokens
