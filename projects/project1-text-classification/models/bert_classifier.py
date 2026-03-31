"""Baseline BERT classifier with configurable pooling and encoder freezing."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BertClassifier"]


class BertClassifier(nn.Module):
    """Pretrained BERT encoder with a simple configurable classification head."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pooling_strategy: str = "mean",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "BertClassifier requires the transformers package."
            ) from error

        try:
            self.encoder = AutoModel.from_pretrained(
                pretrained_model_name,
                local_files_only=True,
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.pooling_strategy = pooling_strategy
        if self.pooling_strategy not in {"mean", "cls"}:
            raise ValueError(
                "pooling_strategy must be either 'mean' or 'cls'."
            )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = int(self.encoder.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(classifier_hidden_dim, num_classes)

    def mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token embeddings with a mask-aware mean."""
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked_embeddings = token_embeddings * mask
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        return masked_embeddings.sum(dim=1) / token_counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if self.pooling_strategy == "cls":
            pooled_output = outputs.last_hidden_state[:, 0]
        else:
            pooled_output = self.mean_pool(
                token_embeddings=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )
        projected_features = self.projection(pooled_output)
        return self.classifier(projected_features)
