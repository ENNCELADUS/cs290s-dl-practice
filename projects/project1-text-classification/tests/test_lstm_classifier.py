"""Tests for the explicit LSTM classifier pipeline."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from config import load_experiment_config
from data import (
    TextClassificationDataset,
    build_text_encoder,
    collate_text_classification_batch,
    load_text_examples,
)
from models import LstmClassifier, build_text_classifier
from train import compute_classification_metrics


def test_load_experiment_config_uses_explicit_lstm_names() -> None:
    experiment_config = load_experiment_config(Path("./configs/lstm_classifier.toml"))

    assert experiment_config.experiment_name == "lstm_classifier"
    assert experiment_config.model.name == "lstm_classifier"
    assert experiment_config.output.checkpoint_dir.name == "lstm_classifier"
    assert experiment_config.output.log_dir.name == "lstm_classifier"


def test_character_encoder_dataset_and_collate_shapes() -> None:
    experiment_config = load_experiment_config(Path("./configs/lstm_classifier.toml"))
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
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_text_classification_batch,
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (4, 16)
    assert batch["labels"].shape == (4,)
    assert batch["lengths"].shape == (4,)


def test_lstm_classifier_forward_output_shape() -> None:
    experiment_config = load_experiment_config(Path("./configs/lstm_classifier.toml"))
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

    assert isinstance(model, LstmClassifier)
    logits = model(
        input_ids=batch["input_ids"],
        lengths=batch["lengths"],
    )
    assert logits.shape == (4, 2)


def test_compute_classification_metrics_returns_valid_scores() -> None:
    import torch

    logits = torch.tensor(
        [
            [0.1, 1.4],
            [1.2, 0.3],
            [0.4, 0.9],
            [0.8, 0.7],
        ]
    )
    labels = torch.tensor([1, 0, 1, 0])

    metrics = compute_classification_metrics(logits=logits, labels=labels)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "macro_f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
