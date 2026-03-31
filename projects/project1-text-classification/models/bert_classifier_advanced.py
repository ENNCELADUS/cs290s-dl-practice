"""Advanced BERT classifier with layer mixing, gated pooling, and LoRA."""

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
    """Pretrained BERT encoder with explainable pooled representations."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.08,
        layer_mix_depth: int = 4,
        enable_bitfit: bool = True,
        enable_layer_norm_tuning: bool = False,
        lora_rank: int = 24,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        lora_target_layers: int = 12,
        lora_output_target_layers: int = 4,
        unfreeze_top_layers: int = 0,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "BertClassifierAdvanced requires the transformers package."
            ) from error

        self.layer_mix_depth = layer_mix_depth
        self.enable_bitfit = enable_bitfit
        self.enable_layer_norm_tuning = enable_layer_norm_tuning
        self.unfreeze_top_layers = unfreeze_top_layers
        try:
            self.encoder = AutoModel.from_pretrained(
                pretrained_model_name,
                local_files_only=True,
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        self._freeze_encoder_parameters()
        self._inject_lora_adapters(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_layers=lora_target_layers,
            output_target_layers=lora_output_target_layers,
        )
        self._enable_low_cost_encoder_tuning()

        hidden_size = int(self.encoder.config.hidden_size)
        self.layer_mix_logits = nn.Parameter(torch.zeros(layer_mix_depth))
        self.attention_pooler = nn.Linear(hidden_size, 1)
        self.pool_gate = nn.Linear(hidden_size * 4, 4)
        self.feature_norm = nn.LayerNorm(hidden_size * 5)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 5, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(classifier_hidden_dim, num_classes)

    def _freeze_encoder_parameters(self) -> None:
        """Freeze pretrained backbone weights before adding LoRA adapters."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if self.unfreeze_top_layers <= 0:
            return
        for layer in self.encoder.encoder.layer[-self.unfreeze_top_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def _inject_lora_adapters(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        target_layers: int,
        output_target_layers: int,
    ) -> None:
        """Attach LoRA modules to selected encoder projections."""
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
            attention_block.key = LoRALinear.from_linear(
                linear=attention_block.key,
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

        selected_output_layers = encoder_layers[
            max(total_layers - output_target_layers, 0) :
        ]
        for layer in selected_output_layers:
            layer.attention.output.dense = LoRALinear.from_linear(
                linear=layer.attention.output.dense,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    def _enable_low_cost_encoder_tuning(self) -> None:
        """Unfreeze only low-cost normalization and bias parameters."""
        for name, parameter in self.encoder.named_parameters():
            is_layer_norm_parameter = "LayerNorm" in name
            is_bias_parameter = name.endswith(".bias")
            if self.enable_layer_norm_tuning and is_layer_norm_parameter:
                parameter.requires_grad = True
            elif self.enable_bitfit and is_bias_parameter:
                parameter.requires_grad = True

    def mix_hidden_layers(
        self,
        hidden_states: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Learn a weighted mixture of the top encoder hidden states."""
        if len(hidden_states) < self.layer_mix_depth:
            raise ValueError("Not enough hidden states available for layer mixing.")
        selected_hidden_states = hidden_states[-self.layer_mix_depth :]
        mix_weights = torch.softmax(self.layer_mix_logits, dim=0).view(-1, 1, 1, 1)
        mixed_hidden_states = torch.zeros_like(selected_hidden_states[0])
        for weight, hidden_state in zip(mix_weights, selected_hidden_states):
            mixed_hidden_states = mixed_hidden_states + (weight * hidden_state)
        return mixed_hidden_states

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

    def max_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token embeddings with a mask-aware max."""
        mask = attention_mask.unsqueeze(-1).bool()
        masked_embeddings = token_embeddings.masked_fill(~mask, torch.finfo(token_embeddings.dtype).min)
        return masked_embeddings.max(dim=1).values

    def pool_features(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse CLS, mean, attention, and max pooling with a learned gate."""
        cls_features = token_embeddings[:, 0]
        mean_features = self.mean_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        attention_features = self.attention_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        max_features = self.max_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        pooled_stack = torch.stack(
            [cls_features, mean_features, attention_features, max_features],
            dim=1,
        )
        gate_inputs = torch.cat(
            [cls_features, mean_features, attention_features, max_features],
            dim=1,
        )
        gate_weights = torch.softmax(self.pool_gate(gate_inputs), dim=1).unsqueeze(-1)
        gated_features = (pooled_stack * gate_weights).sum(dim=1)
        combined_features = torch.cat(
            [
                cls_features,
                mean_features,
                attention_features,
                max_features,
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
        """Encode text, fuse semantic pooling signals, and classify."""
        try:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        except TypeError:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            token_embeddings = outputs.last_hidden_state
        else:
            token_embeddings = self.mix_hidden_layers(hidden_states)
        pooled_features = self.pool_features(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        projected_features = self.projection(pooled_features)
        return self.classifier(projected_features)
