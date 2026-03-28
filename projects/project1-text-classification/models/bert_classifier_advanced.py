"""Advanced BERT classifier with LoRA and semantic pooling fusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["BertClassifierAdvanced", "LoRALinear"]


class LoRALinear(nn.Module):
    """Linear layer augmented with a low-rank adaptation branch."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.scaling = alpha / float(rank)
        self.lora_dropout: nn.Module
        if dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = nn.Identity()
        self.lora_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_linear.out_features, bias=False)
        self.reset_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        """Wrap an existing linear layer with a trainable LoRA update."""
        return cls(
            base_linear=linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def reset_parameters(self) -> None:
        """Initialize the LoRA branch with a zero-impact starting point."""
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection plus a low-rank update."""
        base_output = self.base_linear(inputs)
        lora_output = self.lora_b(self.lora_a(self.lora_dropout(inputs)))
        return base_output + (self.scaling * lora_output)


class BertClassifierAdvanced(nn.Module):
    """Pretrained BERT encoder with LoRA and fused semantic pooling."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.1,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_layers: int = 4,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "BertClassifierAdvanced requires the transformers package."
            ) from error

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self._freeze_encoder_parameters()
        self._inject_lora_adapters(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_layers=lora_target_layers,
        )

        hidden_size = int(self.encoder.config.hidden_size)
        self.attention_pooler = nn.Linear(hidden_size, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 3, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(classifier_hidden_dim, num_classes)

    def _freeze_encoder_parameters(self) -> None:
        """Freeze pretrained backbone weights before adding LoRA adapters."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def _inject_lora_adapters(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        target_layers: int,
    ) -> None:
        """Attach LoRA modules to query/value projections in upper layers."""
        encoder_layers = self.encoder.encoder.layer
        total_layers = len(encoder_layers)
        selected_layers = encoder_layers[max(total_layers - target_layers, 0) :]
        for layer in selected_layers:
            attention_block = layer.attention.self
            attention_block.query = LoRALinear.from_linear(
                linear=attention_block.query,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            attention_block.value = LoRALinear.from_linear(
                linear=attention_block.value,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

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
        """Learn token weights and aggregate a task-focused sentence vector."""
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
        """Fuse CLS, mean, and attention pooling into one representation."""
        cls_features = token_embeddings[:, 0]
        mean_features = self.mean_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        attention_features = self.attention_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        return torch.cat(
            [cls_features, mean_features, attention_features],
            dim=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text, fuse semantic pooling signals, and classify."""
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
