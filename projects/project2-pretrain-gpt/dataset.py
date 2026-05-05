from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from datasets import DatasetDict, DownloadConfig, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    """A next-token language-modeling dataset over contiguous token ids."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.token_ids = token_ids
        self.block_size = block_size
        self.num_blocks = max(0, (len(token_ids) - 1) // block_size)

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(index)
        start = index * self.block_size
        end = start + self.block_size
        input_ids = torch.tensor(self.token_ids[start:end], dtype=torch.long)
        labels = torch.tensor(self.token_ids[start + 1 : end + 1], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


@dataclass(frozen=True)
class DatasetBuildConfig:
    dataset_name: str
    dataset_config_name: str | None
    block_size: int
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    num_preprocessing_workers: int = 1
    local_files_only: bool = False


def _limit_split(split: Any, max_samples: int | None) -> Any:
    if max_samples is None:
        return split
    return split.select(range(min(max_samples, len(split))))


def _tokenize_split(
    split: Any,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    num_preprocessing_workers: int,
) -> list[int]:
    def tokenize_function(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        return tokenizer(examples[text_column], add_special_tokens=True, truncation=False)

    tokenized = split.map(
        tokenize_function,
        batched=True,
        num_proc=num_preprocessing_workers,
        remove_columns=split.column_names,
        desc="Tokenizing text",
    )

    token_ids: list[int] = []
    for input_ids in tokenized["input_ids"]:
        token_ids.extend(input_ids)
    return token_ids


def build_token_block_datasets(
    config: DatasetBuildConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[TokenBlockDataset, TokenBlockDataset]:
    """Load TinyStories-style train/validation splits and pack them into blocks."""

    raw = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        download_config=DownloadConfig(local_files_only=config.local_files_only),
    )
    if not isinstance(raw, DatasetDict) or "train" not in raw or "validation" not in raw:
        raise ValueError("Expected dataset with train and validation splits.")

    splits = DatasetDict(
        train=_limit_split(raw["train"], config.max_train_samples),
        validation=_limit_split(raw["validation"], config.max_validation_samples),
    )
    column_names = splits["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]

    train_ids = _tokenize_split(
        splits["train"],
        tokenizer,
        text_column,
        config.num_preprocessing_workers,
    )
    validation_ids = _tokenize_split(
        splits["validation"],
        tokenizer,
        text_column,
        config.num_preprocessing_workers,
    )

    train_dataset = TokenBlockDataset(train_ids, config.block_size)
    validation_dataset = TokenBlockDataset(validation_ids, config.block_size)
    LOGGER.info(
        "Built token blocks: train=%s validation=%s block_size=%s",
        len(train_dataset),
        len(validation_dataset),
        config.block_size,
    )
    return train_dataset, validation_dataset
