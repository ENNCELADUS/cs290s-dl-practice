"""Baseline BERT classifier with configurable pooling and encoder freezing."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BertClassifier"]


class BertClassifier(nn.Module):
    """Pretrained BERT encoder with a compact fused-pooling classification head."""

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
        self.attention_pooler = nn.Linear(hidden_size, 1)
        self.pool_gate = nn.Linear(hidden_size * 3, 3)
        self.feature_norm = nn.LayerNorm(hidden_size * 4)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 4, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(classifier_hidden_dim, num_classes)
        self._reset_head_parameters()

    def _reset_head_parameters(self) -> None:
        """Initialize pooling layers to stable, near-uniform defaults."""
        nn.init.zeros_(self.attention_pooler.weight)
        nn.init.zeros_(self.attention_pooler.bias)
        nn.init.zeros_(self.pool_gate.weight)
        nn.init.zeros_(self.pool_gate.bias)

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

    def attention_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Learn token weights and aggregate a task-aware sentence vector."""
        attention_scores = self.attention_pooler(token_embeddings).squeeze(-1)
        mask_value = torch.finfo(token_embeddings.dtype).min
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0,
            mask_value,
        )
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        return (token_embeddings * attention_weights).sum(dim=1)

    def pool_features(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse anchor, complementary, and attention pooling into one feature vector."""
        cls_features = token_embeddings[:, 0]
        mean_features = self.mean_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        attention_features = self.attention_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        if self.pooling_strategy == "cls":
            anchor_features = cls_features
            complementary_features = mean_features
        else:
            anchor_features = mean_features
            complementary_features = cls_features
        pooled_stack = torch.stack(
            [anchor_features, complementary_features, attention_features],
            dim=1,
        )
        gate_inputs = torch.cat(
            [anchor_features, complementary_features, attention_features],
            dim=1,
        )
        gate_weights = torch.softmax(self.pool_gate(gate_inputs), dim=1).unsqueeze(-1)
        gated_features = (pooled_stack * gate_weights).sum(dim=1)
        combined_features = torch.cat(
            [
                anchor_features,
                complementary_features,
                attention_features,
                gated_features,
            ],
            dim=1,
        )
        return self.feature_norm(combined_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_features = self.pool_features(
            token_embeddings=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )
        projected_features = self.projection(pooled_features)
        return self.classifier(projected_features)
