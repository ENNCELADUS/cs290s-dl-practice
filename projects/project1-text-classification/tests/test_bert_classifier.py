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
from models import BertClassifier, BertClassifierAdvanced, build_text_classifier
from models.bert_classifier_advanced import LoRALinear


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


class _FakeSelfAttention(torch.nn.Module):
    """Fake self-attention block with named projections."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)


class _FakeAttention(torch.nn.Module):
    """Fake attention container matching Hugging Face naming."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self = _FakeSelfAttention(hidden_size=hidden_size)


class _FakeLayer(torch.nn.Module):
    """Fake transformer layer for LoRA replacement tests."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = _FakeAttention(hidden_size=hidden_size)


class _FakeEncoder(torch.nn.Module):
    """Fake encoder stack exposing a Hugging Face-style layer list."""

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList(
            [_FakeLayer(hidden_size=hidden_size) for _ in range(num_layers)]
        )


class _FakeAutoModel(torch.nn.Module):
    """Minimal encoder stub that exposes hidden states and transformer layers."""

    def __init__(self, hidden_size: int = 12, num_layers: int = 4) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.encoder = _FakeEncoder(hidden_size=hidden_size, num_layers=num_layers)

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


def _build_transformer_batch() -> dict[str, torch.Tensor]:
    """Build a small transformer-style batch for model tests."""
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
    return collate_text_classification_batch([dataset[index] for index in range(4)])


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
    batch = _build_transformer_batch()

    assert batch["input_ids"].shape == (4, 16)
    assert batch["attention_mask"].shape == (4, 16)
    assert batch["labels"].shape == (4,)


def test_bert_classifier_forward_output_shape_for_mean_pooling() -> None:
    experiment_config = load_experiment_config(Path("./configs/bert_classifier.toml"))
    batch = _build_transformer_batch()

    model = build_text_classifier(
        model_config=experiment_config.model,
        vocabulary_size=21128,
        pad_id=0,
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


def test_bert_classifier_uses_masked_mean_pooling() -> None:
    model = BertClassifier(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        num_classes=2,
        classifier_hidden_dim=12,
        dropout=0.0,
    )
    input_ids = torch.tensor(
        [
            [10, 20, 0, 0],
            [2, 4, 6, 8],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    pooled_output = model.mean_pool(
        token_embeddings=model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state,
        attention_mask=attention_mask,
    )

    expected = torch.tensor(
        [
            [15.0] * 12,
            [4.0] * 12,
        ]
    )
    assert torch.allclose(pooled_output, expected)


def test_build_text_classifier_can_construct_advanced_bert_variant() -> None:
    advanced_model_config = types.SimpleNamespace(
        name="bert_classifier_advanced",
        parameters={
            "pretrained_model_name": "hfl/chinese-roberta-wwm-ext",
            "num_classes": 2,
            "classifier_hidden_dim": 256,
            "dropout": 0.1,
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "lora_dropout": 0.05,
            "lora_target_layers": 2,
        },
    )

    model = build_text_classifier(
        model_config=advanced_model_config,
        vocabulary_size=None,
        pad_id=None,
    )

    assert isinstance(model, BertClassifierAdvanced)


def test_advanced_bert_classifier_forward_output_shape() -> None:
    batch = _build_transformer_batch()
    model = BertClassifierAdvanced(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        num_classes=2,
        classifier_hidden_dim=24,
        dropout=0.0,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_layers=2,
    )

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    assert logits.shape == (4, 2)


def test_advanced_bert_classifier_replaces_query_and_value_with_lora_on_top_layers() -> None:
    model = BertClassifierAdvanced(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        num_classes=2,
        classifier_hidden_dim=24,
        dropout=0.0,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_layers=2,
    )
    encoder_layers = model.encoder.encoder.layer

    assert isinstance(encoder_layers[0].attention.self.query, torch.nn.Linear)
    assert isinstance(encoder_layers[0].attention.self.value, torch.nn.Linear)
    assert isinstance(encoder_layers[1].attention.self.query, torch.nn.Linear)
    assert isinstance(encoder_layers[1].attention.self.value, torch.nn.Linear)
    assert isinstance(encoder_layers[2].attention.self.query, LoRALinear)
    assert isinstance(encoder_layers[2].attention.self.value, LoRALinear)
    assert isinstance(encoder_layers[3].attention.self.query, LoRALinear)
    assert isinstance(encoder_layers[3].attention.self.value, LoRALinear)


def test_advanced_bert_classifier_fuses_cls_mean_and_attention_pooling() -> None:
    model = BertClassifierAdvanced(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        num_classes=2,
        classifier_hidden_dim=24,
        dropout=0.0,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_layers=2,
    )
    with torch.no_grad():
        model.attention_pooler.weight.zero_()
        model.attention_pooler.bias.zero_()

    token_embeddings = torch.tensor(
        [
            [
                [1.0] * 12,
                [3.0] * 12,
                [5.0] * 12,
                [99.0] * 12,
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    pooled_features = model.pool_features(
        token_embeddings=token_embeddings,
        attention_mask=attention_mask,
    )

    assert pooled_features.shape == (1, 36)
    assert torch.allclose(pooled_features[:, :12], torch.tensor([[1.0] * 12]))
    assert torch.allclose(pooled_features[:, 12:24], torch.tensor([[3.0] * 12]))
    assert torch.allclose(pooled_features[:, 24:], torch.tensor([[3.0] * 12]))
