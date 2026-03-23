"""Public model interfaces and model factory for text classification."""

from __future__ import annotations

from torch import nn

from config import ModelConfig
from models.bert_classifier import BertClassifier
from models.lstm_classifier import LstmClassifier

__all__ = [
    "BertClassifier",
    "LstmClassifier",
    "build_text_classifier",
]


def build_text_classifier(
    model_config: ModelConfig,
    vocabulary_size: int | None,
    pad_id: int | None,
) -> nn.Module:
    """Instantiate the configured classifier."""
    parameters = model_config.parameters

    if model_config.name == "lstm_classifier":
        if vocabulary_size is None or pad_id is None:
            raise ValueError(
                "lstm_classifier requires vocabulary_size and pad_id from the text encoder."
            )
        return LstmClassifier(
            vocabulary_size=vocabulary_size,
            embedding_dim=int(parameters["embedding_dim"]),
            hidden_dim=int(parameters["hidden_dim"]),
            num_classes=int(parameters["num_classes"]),
            pad_id=pad_id,
            num_layers=int(parameters.get("num_layers", 1)),
            dropout=float(parameters.get("dropout", 0.3)),
        )

    if model_config.name == "bert_classifier":
        return BertClassifier(
            pretrained_model_name=str(parameters["pretrained_model_name"]),
            num_classes=int(parameters["num_classes"]),
            classifier_hidden_dim=int(parameters.get("classifier_hidden_dim", 256)),
            dropout=float(parameters.get("dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model name: {model_config.name}")
