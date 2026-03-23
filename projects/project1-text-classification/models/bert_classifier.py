"""BERT-style classifier for transformer-based text classification."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BertClassifier"]


class BertClassifier(nn.Module):
    """Pretrained BERT encoder with CLS pooling and a 2-layer MLP head."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "BertClassifier requires the transformers package."
            ) from error

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(classifier_hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        projected_features = self.projection(pooled_output)
        return self.classifier(projected_features)
