"""Tests for the transformer-based text classification pipeline."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from config import load_experiment_config
from data import (
    TextClassificationDataset,
    build_text_encoder,
    collate_text_classification_batch,
    load_text_examples,
)
from models import BertClassifier, build_text_classifier


class _FakeTokenizer:
    """Minimal tokenizer stub for transformer-path tests."""

    vocab_size = 21128
    pad_token_id = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str) -> "_FakeTokenizer":
        del pretrained_model_name
        return cls()

    def __call__(
        self,
        text: str,
        truncation: bool,
        padding: str,
        max_length: int,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        del truncation, padding, return_attention_mask
        token_count = min(len(text), max_length - 2)
        input_ids = [101] + list(range(1000, 1000 + token_count)) + [102]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _FakeAutoModel(torch.nn.Module):
    """Minimal encoder stub that exposes hidden states and config."""

    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str) -> "_FakeAutoModel":
        del pretrained_model_name
        return cls()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> types.SimpleNamespace:
        del attention_mask
        hidden_states = input_ids.unsqueeze(-1).float().repeat(1, 1, 12)
        return types.SimpleNamespace(last_hidden_state=hidden_states)


@pytest.fixture(autouse=True)
def patch_transformers_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a local transformers stub for model and tokenizer imports."""
    fake_module = types.ModuleType("transformers")
    fake_module.AutoModel = _FakeAutoModel
    fake_module.AutoTokenizer = _FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_module)


def test_load_experiment_config_uses_explicit_bert_names() -> None:
    experiment_config = load_experiment_config(Path("./configs/bert_classifier.toml"))

    assert experiment_config.experiment_name == "bert_classifier"
    assert experiment_config.model.name == "bert_classifier"
    assert experiment_config.output.checkpoint_dir.name == "bert_classifier"
    assert experiment_config.output.log_dir.name == "bert_classifier"


def test_transformer_encoder_dataset_and_collate_shapes() -> None:
    experiment_config = load_experiment_config(Path("./configs/bert_classifier.toml"))
    train_examples = load_text_examples(
        data_config=experiment_config.data,
        csv_path=experiment_config.data.train_csv,
    )[:4]
    text_encoder = build_text_encoder(
        data_config=experiment_config.data,
        training_examples=train_examples,
    )
    dataset = TextClassificationDataset(
        examples=train_examples,
        text_encoder=text_encoder,
        max_length=16,
    )

    batch = collate_text_classification_batch([dataset[index] for index in range(4)])

    assert batch["input_ids"].shape == (4, 16)
    assert batch["attention_mask"].shape == (4, 16)
    assert batch["labels"].shape == (4,)


def test_bert_classifier_forward_output_shape_for_cls_pooling() -> None:
    experiment_config = load_experiment_config(Path("./configs/bert_classifier.toml"))
    train_examples = load_text_examples(
        data_config=experiment_config.data,
        csv_path=experiment_config.data.train_csv,
    )[:4]
    text_encoder = build_text_encoder(
        data_config=experiment_config.data,
        training_examples=train_examples,
    )
    dataset = TextClassificationDataset(
        examples=train_examples,
        text_encoder=text_encoder,
        max_length=16,
    )
    batch = collate_text_classification_batch([dataset[index] for index in range(4)])

    model = build_text_classifier(
        model_config=experiment_config.model,
        vocabulary_size=text_encoder.vocabulary_size,
        pad_id=text_encoder.pad_id,
    )

    assert isinstance(model, BertClassifier)
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    assert logits.shape == (4, 2)


def test_bert_classifier_uses_two_layer_mlp_head() -> None:
    experiment_config = load_experiment_config(Path("./configs/bert_classifier.toml"))
    model = build_text_classifier(
        model_config=experiment_config.model,
        vocabulary_size=None,
        pad_id=None,
    )

    assert isinstance(model, BertClassifier)
    assert isinstance(model.projection, torch.nn.Sequential)
    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.classifier.in_features == 256
    assert model.classifier.out_features == 2
