"""Autoresearch training script for the Project 1 advanced BERT experiment."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm

from prepare import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PRETRAINED_MODEL_NAME,
    evaluate_model,
    make_dataloaders,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters and experiment settings
# Edit these directly. This file is the only experiment surface.
# ---------------------------------------------------------------------------

PRETRAINED_MODEL_NAME = DEFAULT_PRETRAINED_MODEL_NAME
NUM_CLASSES = DEFAULT_NUM_CLASSES
MAX_LENGTH = DEFAULT_MAX_LENGTH

# Architecture
CLASSIFIER_HIDDEN_DIM = 256
DROPOUT = 0.1
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
LORA_TARGET_LAYERS = 4

# Optimization
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 3
VAL_STEPS = 200
SEED = 42

# Outputs
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


def configure_logging() -> None:
    """Configure plain-text logging."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class BertClassifierExperiment(nn.Module):
    """Chinese RoBERTa classifier with LoRA and fused pooling."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "BertClassifierExperiment requires the transformers package."
            ) from error

        self.encoder = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self._freeze_encoder_parameters()
        self._inject_lora_adapters(
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
            target_layers=LORA_TARGET_LAYERS,
        )

        hidden_size = int(self.encoder.config.hidden_size)
        self.attention_pooler = nn.Linear(hidden_size, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 3, CLASSIFIER_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(p=DROPOUT),
        )
        self.classifier = nn.Linear(CLASSIFIER_HIDDEN_DIM, NUM_CLASSES)

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


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a batch dictionary onto the training device."""
    return {feature_name: tensor.to(device) for feature_name, tensor in batch.items()}


def save_checkpoint(
    model: torch.nn.Module,
    metrics: dict[str, float],
) -> None:
    """Save the best model checkpoint and experiment settings."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "hyperparameters": {
                "pretrained_model_name": PRETRAINED_MODEL_NAME,
                "max_length": MAX_LENGTH,
                "classifier_hidden_dim": CLASSIFIER_HIDDEN_DIM,
                "dropout": DROPOUT,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_layers": LORA_TARGET_LAYERS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "val_steps": VAL_STEPS,
                "seed": SEED,
            },
        },
        CHECKPOINT_PATH,
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train() -> None:
    """Run one autoresearch experiment."""
    configure_logging()
    set_seed(SEED)

    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader = make_dataloaders(
        batch_size=BATCH_SIZE,
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        max_length=MAX_LENGTH,
        seed=SEED,
    )

    model = BertClassifierExperiment().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    LOGGER.info("Device: %s", device)
    LOGGER.info("Train batches: %s", len(train_loader))
    LOGGER.info("Validation batches: %s", len(val_loader))
    LOGGER.info("Trainable params: %.2fM", count_trainable_parameters(model) / 1_000_000)

    best_macro_f1 = float("-inf")
    best_metrics: dict[str, float] | None = None
    total_steps = 0
    training_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )
        for batch in progress_bar:
            device_batch = move_batch_to_device(batch=batch, device=device)
            labels = device_batch["labels"]
            logits = model(
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
            )
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if total_steps % VAL_STEPS == 0:
                val_metrics = evaluate_model(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                )
                LOGGER.info(
                    (
                        "step=%s val_loss=%.4f accuracy=%.4f "
                        "f1=%.4f macro_f1=%.4f"
                    ),
                    total_steps,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["f1"],
                    val_metrics["macro_f1"],
                )
                if val_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = val_metrics["macro_f1"]
                    best_metrics = val_metrics
                    save_checkpoint(model=model, metrics=val_metrics)

        val_metrics = evaluate_model(model=model, val_loader=val_loader, device=device)
        LOGGER.info(
            (
                "epoch=%s val_loss=%.4f accuracy=%.4f "
                "f1=%.4f macro_f1=%.4f"
            ),
            epoch + 1,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["macro_f1"],
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            save_checkpoint(model=model, metrics=val_metrics)

    if best_metrics is None:
        best_metrics = evaluate_model(model=model, val_loader=val_loader, device=device)
        save_checkpoint(model=model, metrics=best_metrics)

    training_seconds = time.time() - training_start_time
    total_seconds = time.time() - total_start_time
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    LOGGER.info("---")
    LOGGER.info("macro_f1: %.6f", best_metrics["macro_f1"])
    LOGGER.info("accuracy: %.6f", best_metrics["accuracy"])
    LOGGER.info("precision: %.6f", best_metrics["precision"])
    LOGGER.info("recall: %.6f", best_metrics["recall"])
    LOGGER.info("f1: %.6f", best_metrics["f1"])
    LOGGER.info("val_loss: %.6f", best_metrics["loss"])
    LOGGER.info("training_seconds: %.1f", training_seconds)
    LOGGER.info("total_seconds: %.1f", total_seconds)
    LOGGER.info("peak_vram_mb: %.1f", peak_vram_mb)
    LOGGER.info(
        "trainable_params_M: %.3f",
        count_trainable_parameters(model) / 1_000_000,
    )
    LOGGER.info("classifier_hidden_dim: %s", CLASSIFIER_HIDDEN_DIM)
    LOGGER.info("lora_rank: %s", LORA_RANK)
    LOGGER.info("lora_alpha: %.1f", LORA_ALPHA)
    LOGGER.info("lora_dropout: %.3f", LORA_DROPOUT)
    LOGGER.info("lora_target_layers: %s", LORA_TARGET_LAYERS)
    LOGGER.info("batch_size: %s", BATCH_SIZE)
    LOGGER.info("learning_rate: %.8f", LEARNING_RATE)
    LOGGER.info("weight_decay: %.6f", WEIGHT_DECAY)


if __name__ == "__main__":
    train()
