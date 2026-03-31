"""Public model interfaces and model factory for text classification."""

from __future__ import annotations

from torch import nn

from config import ModelConfig
from models.bert_classifier import BertClassifier
from models.bert_classifier_advanced import BertClassifierAdvanced
from models.lstm_classifier import LstmClassifier

__all__ = [
    "BertClassifier",
    "BertClassifierAdvanced",
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
            freeze_encoder=bool(parameters.get("freeze_encoder", False)),
            pooling_strategy=str(parameters.get("pooling_strategy", "mean")),
        )

    if model_config.name == "bert_classifier_advanced":
        return BertClassifierAdvanced(
            pretrained_model_name=str(parameters["pretrained_model_name"]),
            num_classes=int(parameters["num_classes"]),
            classifier_hidden_dim=int(parameters.get("classifier_hidden_dim", 512)),
            dropout=float(parameters.get("dropout", 0.08)),
            layer_mix_depth=int(parameters.get("layer_mix_depth", 4)),
            enable_bitfit=bool(parameters.get("enable_bitfit", True)),
            enable_layer_norm_tuning=bool(
                parameters.get("enable_layer_norm_tuning", False)
            ),
            lora_rank=int(parameters.get("lora_rank", 24)),
            lora_alpha=float(parameters.get("lora_alpha", 32.0)),
            lora_dropout=float(parameters.get("lora_dropout", 0.0)),
            lora_target_layers=int(parameters.get("lora_target_layers", 12)),
            lora_output_target_layers=int(
                parameters.get("lora_output_target_layers", 4)
            ),
            unfreeze_top_layers=int(parameters.get("unfreeze_top_layers", 0)),
        )

    raise ValueError(f"Unsupported model name: {model_config.name}")
