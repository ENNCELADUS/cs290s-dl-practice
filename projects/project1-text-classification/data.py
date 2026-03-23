"""Dataset and text-encoding utilities for text classification experiments."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import torch
from torch.utils.data import Dataset

from config import DataConfig

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

__all__ = [
    "CharacterTextEncoder",
    "CharacterVocabulary",
    "TextClassificationDataset",
    "TextEncoder",
    "TextExample",
    "TransformerTextEncoder",
    "build_text_encoder",
    "collate_text_classification_batch",
    "load_text_examples",
]


@dataclass(frozen=True)
class TextExample:
    """One labeled text example from the CSV dataset."""

    text: str
    label: int


class TextEncoder(Protocol):
    """Protocol for model-specific text encoders."""

    @property
    def vocabulary_size(self) -> int | None:
        """Return the vocabulary size when applicable."""

    @property
    def pad_id(self) -> int | None:
        """Return the padding id when applicable."""

    def encode(self, text: str, max_length: int) -> dict[str, list[int] | int]:
        """Encode raw text into model-ready features."""

    def export_state(self) -> dict[str, object]:
        """Return serializable encoder metadata for checkpoints."""


class CharacterVocabulary:
    """Character-level vocabulary with explicit special-token handling."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self._token_to_id = token_to_id

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        min_frequency: int = 1,
    ) -> "CharacterVocabulary":
        """Build a vocabulary from training texts."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)

        token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }
        for token, frequency in sorted(counter.items()):
            if frequency >= min_frequency and token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    @property
    def pad_id(self) -> int:
        """Return the padding token id."""
        return self._token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        """Return the unknown token id."""
        return self._token_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        """Return the vocabulary size."""
        return len(self._token_to_id)

    def encode(self, text: str, max_length: int) -> tuple[list[int], int]:
        """Convert raw text to a padded character-id sequence."""
        characters = list(text)
        if not characters:
            characters = [UNK_TOKEN]

        truncated_characters = characters[:max_length]
        effective_length = max(1, len(truncated_characters))
        encoded_characters = [
            self._token_to_id.get(character, self.unk_id)
            for character in truncated_characters
        ]

        if len(encoded_characters) < max_length:
            encoded_characters.extend(
                [self.pad_id] * (max_length - len(encoded_characters))
            )

        return encoded_characters, effective_length

    def export_state(self) -> dict[str, object]:
        """Return serializable vocabulary state."""
        return {"token_to_id": dict(self._token_to_id)}


class CharacterTextEncoder:
    """Character-level encoder for the BiLSTM classifier."""

    def __init__(self, vocabulary: CharacterVocabulary) -> None:
        self._vocabulary = vocabulary

    @property
    def vocabulary_size(self) -> int | None:
        return self._vocabulary.size

    @property
    def pad_id(self) -> int | None:
        return self._vocabulary.pad_id

    def encode(self, text: str, max_length: int) -> dict[str, list[int] | int]:
        input_ids, length = self._vocabulary.encode(text=text, max_length=max_length)
        return {
            "input_ids": input_ids,
            "lengths": length,
        }

    def export_state(self) -> dict[str, object]:
        return {
            "encoder_type": "character",
            "vocabulary": self._vocabulary.export_state(),
        }


class TransformerTextEncoder:
    """Transformer tokenizer wrapper for BERT-style classifiers."""

    def __init__(self, pretrained_model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "TransformerTextEncoder requires the transformers package."
            ) from error

        self._pretrained_model_name = pretrained_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    @property
    def vocabulary_size(self) -> int | None:
        return int(self._tokenizer.vocab_size)

    @property
    def pad_id(self) -> int | None:
        return int(self._tokenizer.pad_token_id)

    def encode(self, text: str, max_length: int) -> dict[str, list[int] | int]:
        encoded = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
        }

    def export_state(self) -> dict[str, object]:
        return {
            "encoder_type": "transformer",
            "pretrained_model_name": self._pretrained_model_name,
            "pad_id": self.pad_id,
        }


def load_text_examples(data_config: DataConfig, csv_path: Path) -> list[TextExample]:
    """Load labeled examples from a CSV file."""
    examples: list[TextExample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            examples.append(
                TextExample(
                    text=row[data_config.text_field],
                    label=int(row[data_config.label_field]),
                )
            )
    return examples


def build_text_encoder(
    data_config: DataConfig,
    training_examples: Iterable[TextExample],
) -> TextEncoder:
    """Build the configured text encoder from training data or a pretrained tokenizer."""
    if data_config.encoding == "character":
        vocabulary = CharacterVocabulary.from_texts(
            texts=(example.text for example in training_examples),
            min_frequency=data_config.min_frequency,
        )
        return CharacterTextEncoder(vocabulary=vocabulary)

    if data_config.encoding == "transformer":
        if data_config.pretrained_model_name is None:
            raise ValueError(
                "Transformer encoding requires data.pretrained_model_name in config."
            )
        return TransformerTextEncoder(
            pretrained_model_name=data_config.pretrained_model_name
        )

    raise ValueError(f"Unsupported data encoding: {data_config.encoding}")


class TextClassificationDataset(Dataset):
    """Dataset that emits encoded text features and labels."""

    def __init__(
        self,
        examples: list[TextExample],
        text_encoder: TextEncoder,
        max_length: int,
    ) -> None:
        self._examples = examples
        self._text_encoder = text_encoder
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self._examples[index]
        encoded_features = self._text_encoder.encode(
            text=example.text,
            max_length=self._max_length,
        )

        tensor_features = {
            feature_name: torch.tensor(feature_value, dtype=torch.long)
            for feature_name, feature_value in encoded_features.items()
        }
        tensor_features["labels"] = torch.tensor(example.label, dtype=torch.long)
        return tensor_features


def collate_text_classification_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack a batch of encoded examples regardless of model type."""
    feature_names = batch[0].keys()
    return {
        feature_name: torch.stack(
            [example[feature_name] for example in batch],
            dim=0,
        )
        for feature_name in feature_names
    }
