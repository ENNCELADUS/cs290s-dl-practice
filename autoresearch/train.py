"""Autoresearch training and Optuna sweep entry point for Project 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
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

PRETRAINED_MODEL_NAME = DEFAULT_PRETRAINED_MODEL_NAME
NUM_CLASSES = DEFAULT_NUM_CLASSES
MAX_LENGTH = DEFAULT_MAX_LENGTH

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
RESULTS_PATH = Path("results.tsv")
OPTUNA_TRIALS_PATH = Path("optuna_trials.jsonl")
OPTUNA_BEST_PATH = Path("optuna_best.json")
OPTUNA_STORAGE_PATH = Path("optuna_study.db")


@dataclass(frozen=True)
class ExperimentConfig:
    """All tunable architecture and optimization settings for one run."""

    classifier_hidden_dim: int = 512
    dropout: float = 0.08
    layer_mix_depth: int = 4
    enable_bitfit: bool = True
    enable_layer_norm_tuning: bool = False
    lora_rank: int = 24
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_layers: int = 12
    lora_output_target_layers: int = 4
    unfreeze_top_layers: int = 0
    batch_size: int = 12
    learning_rate: float = 5.1261745611921186e-05
    encoder_learning_rate: float = 3.9139370975328926e-05
    weight_decay: float = 0.0016864974751216414
    epochs: int = 4
    seed: int = 42
    warmup_ratio: float = 0.1499362438641551
    final_lr_scale: float = 0.05


DEFAULT_CONFIG = ExperimentConfig()


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
    """Chinese RoBERTa classifier with explainable pooled representations."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "BertClassifierExperiment requires the transformers package."
            ) from error

        self.encoder = AutoModel.from_pretrained(
            PRETRAINED_MODEL_NAME,
            local_files_only=True,
        )
        self._freeze_encoder_parameters()
        self._inject_lora_adapters()
        self._enable_low_cost_encoder_tuning()

        hidden_size = int(self.encoder.config.hidden_size)
        self.layer_mix_logits = nn.Parameter(torch.zeros(config.layer_mix_depth))
        self.attention_pooler = nn.Linear(hidden_size, 1)
        self.pool_gate = nn.Linear(hidden_size * 4, 4)
        self.feature_norm = nn.LayerNorm(hidden_size * 5)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 5, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
        )
        self.classifier = nn.Linear(config.classifier_hidden_dim, NUM_CLASSES)

    def _freeze_encoder_parameters(self) -> None:
        """Freeze the backbone, then optionally unfreeze the top layers."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if self.config.unfreeze_top_layers <= 0:
            return
        for layer in self.encoder.encoder.layer[-self.config.unfreeze_top_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def _inject_lora_adapters(self) -> None:
        """Attach LoRA modules to selected encoder projections."""
        encoder_layers = self.encoder.encoder.layer
        total_layers = len(encoder_layers)
        selected_layers = encoder_layers[
            max(total_layers - self.config.lora_target_layers, 0) :
        ]
        for layer in selected_layers:
            attention_block = layer.attention.self
            attention_block.query = LoRALinear.from_linear(
                linear=attention_block.query,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )
            attention_block.key = LoRALinear.from_linear(
                linear=attention_block.key,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )
            attention_block.value = LoRALinear.from_linear(
                linear=attention_block.value,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )

        selected_output_layers = encoder_layers[
            max(total_layers - self.config.lora_output_target_layers, 0) :
        ]
        for layer in selected_output_layers:
            layer.attention.output.dense = LoRALinear.from_linear(
                linear=layer.attention.output.dense,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )

    def _enable_low_cost_encoder_tuning(self) -> None:
        """Unfreeze only low-cost normalization and bias parameters."""
        for name, parameter in self.encoder.named_parameters():
            is_layer_norm_parameter = "LayerNorm" in name
            is_bias_parameter = name.endswith(".bias")
            if self.config.enable_layer_norm_tuning and is_layer_norm_parameter:
                parameter.requires_grad = True
            elif self.config.enable_bitfit and is_bias_parameter:
                parameter.requires_grad = True

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
        """Pool token embeddings with a mask-aware max operation."""
        mask = attention_mask.unsqueeze(-1).bool()
        mask_value = torch.finfo(token_embeddings.dtype).min
        masked_embeddings = token_embeddings.masked_fill(~mask, mask_value)
        return masked_embeddings.max(dim=1).values

    def pool_features(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse CLS, mean, attention, and max pooling into one representation."""
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
        stacked_features = torch.stack(
            [cls_features, mean_features, attention_features, max_features],
            dim=1,
        )
        pooled_features = stacked_features.flatten(start_dim=1)
        gate_weights = torch.softmax(self.pool_gate(pooled_features), dim=1)
        gated_summary = (
            stacked_features * gate_weights.unsqueeze(-1)
        ).sum(dim=1)
        fused_features = torch.cat([pooled_features, gated_summary], dim=1)
        return self.feature_norm(fused_features)

    def mix_hidden_layers(
        self,
        hidden_states: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Learn a weighted mixture of the top encoder hidden states."""
        selected_hidden_states = hidden_states[-self.config.layer_mix_depth :]
        mix_weights = torch.softmax(self.layer_mix_logits, dim=0)
        mixed_hidden_states = torch.zeros_like(selected_hidden_states[0])
        for weight, hidden_state in zip(mix_weights, selected_hidden_states):
            mixed_hidden_states = mixed_hidden_states + (weight * hidden_state)
        return mixed_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text, fuse semantic pooling signals, and classify."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Encoder hidden states are required for layer mixing.")
        token_embeddings = self.mix_hidden_layers(outputs.hidden_states)
        pooled_features = self.pool_features(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        projected_features = self.projection(pooled_features)
        return self.classifier(projected_features)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a batch dictionary onto the training device."""
    return {name: tensor.to(device) for name, tensor in batch.items()}


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def save_checkpoint(
    model: torch.nn.Module,
    metrics: dict[str, float],
    config: ExperimentConfig,
    checkpoint_path: Path,
) -> None:
    """Save the best model checkpoint and experiment settings."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "hyperparameters": {
                "pretrained_model_name": PRETRAINED_MODEL_NAME,
                "max_length": MAX_LENGTH,
                **asdict(config),
            },
        },
        checkpoint_path,
    )


def split_parameter_groups(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Separate encoder-owned parameters from classifier-head parameters."""
    encoder_parameters: list[torch.nn.Parameter] = []
    adapter_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_parameters.append(parameter)
        else:
            adapter_parameters.append(parameter)
    return encoder_parameters, adapter_parameters


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_training_steps: int,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a warmup-plus-cosine decay schedule."""
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_training_steps <= warmup_steps:
            return 1.0
        progress = (current_step - warmup_steps) / float(
            total_training_steps - warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.final_lr_scale + (
            (1.0 - config.final_lr_scale) * cosine_decay
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_training(
    config: ExperimentConfig,
    checkpoint_path: Path,
    save_best: bool,
    trial: object | None = None,
) -> dict[str, float]:
    """Run one experiment and optionally report progress to Optuna."""
    set_seed(config.seed)
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader = make_dataloaders(
        batch_size=config.batch_size,
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        max_length=MAX_LENGTH,
        seed=config.seed,
    )
    val_steps = max(1, len(train_loader) // 2)
    model = BertClassifierExperiment(config=config).to(device)
    encoder_parameters, adapter_parameters = split_parameter_groups(model=model)
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": config.encoder_learning_rate},
            {"params": adapter_parameters, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    total_training_steps = config.epochs * len(train_loader)
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_training_steps=total_training_steps,
        config=config,
    )

    LOGGER.info("Device: %s", device)
    LOGGER.info("Train batches: %s", len(train_loader))
    LOGGER.info("Validation batches: %s", len(val_loader))
    LOGGER.info(
        "Trainable params: %.2fM",
        count_trainable_parameters(model) / 1_000_000,
    )

    best_macro_f1 = float("-inf")
    best_metrics: dict[str, float] | None = None
    total_steps = 0
    evaluation_index = 0
    training_start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch in progress_bar:
            device_batch = move_batch_to_device(batch=batch, device=device)
            logits = model(
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
            )
            loss = F.cross_entropy(logits, device_batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_steps += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if total_steps % val_steps == 0:
                evaluation_index += 1
                val_metrics = evaluate_model(model=model, val_loader=val_loader, device=device)
                LOGGER.info(
                    "step=%s val_loss=%.4f accuracy=%.4f f1=%.4f macro_f1=%.4f",
                    total_steps,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["f1"],
                    val_metrics["macro_f1"],
                )
                if val_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = val_metrics["macro_f1"]
                    best_metrics = val_metrics
                    if save_best:
                        save_checkpoint(
                            model=model,
                            metrics=val_metrics,
                            config=config,
                            checkpoint_path=checkpoint_path,
                        )
                if trial is not None:
                    trial.report(best_macro_f1, step=evaluation_index)
                    if trial.should_prune():
                        raise prune_trial(f"Pruned at step {total_steps}")

        evaluation_index += 1
        val_metrics = evaluate_model(model=model, val_loader=val_loader, device=device)
        LOGGER.info(
            "epoch=%s val_loss=%.4f accuracy=%.4f f1=%.4f macro_f1=%.4f",
            epoch + 1,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["macro_f1"],
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            if save_best:
                save_checkpoint(
                    model=model,
                    metrics=val_metrics,
                    config=config,
                    checkpoint_path=checkpoint_path,
                )
        if trial is not None:
            trial.report(best_macro_f1, step=evaluation_index)
            if trial.should_prune():
                raise prune_trial(f"Pruned after epoch {epoch + 1}")

    if best_metrics is None:
        best_metrics = evaluate_model(model=model, val_loader=val_loader, device=device)
        if save_best:
            save_checkpoint(
                model=model,
                metrics=best_metrics,
                config=config,
                checkpoint_path=checkpoint_path,
            )

    training_seconds = time.time() - training_start_time
    total_seconds = time.time() - total_start_time
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    summary = {
        "macro_f1": best_metrics["macro_f1"],
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "loss": best_metrics["loss"],
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "peak_vram_mb": peak_vram_mb,
        "trainable_params_M": count_trainable_parameters(model) / 1_000_000,
    }

    LOGGER.info("---")
    LOGGER.info("macro_f1: %.6f", summary["macro_f1"])
    LOGGER.info("accuracy: %.6f", summary["accuracy"])
    LOGGER.info("precision: %.6f", summary["precision"])
    LOGGER.info("recall: %.6f", summary["recall"])
    LOGGER.info("f1: %.6f", summary["f1"])
    LOGGER.info("val_loss: %.6f", summary["loss"])
    LOGGER.info("training_seconds: %.1f", summary["training_seconds"])
    LOGGER.info("total_seconds: %.1f", summary["total_seconds"])
    LOGGER.info("peak_vram_mb: %.1f", summary["peak_vram_mb"])
    LOGGER.info("trainable_params_M: %.3f", summary["trainable_params_M"])
    for key, value in asdict(config).items():
        LOGGER.info("%s: %s", key, value)

    return summary


def append_optuna_result(
    trial_number: int,
    value: float,
    config: ExperimentConfig,
    state: str,
) -> None:
    """Append one Optuna trial record to disk."""
    OPTUNA_TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_number": trial_number,
        "value": value,
        "state": state,
        "config": asdict(config),
    }
    with OPTUNA_TRIALS_PATH.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def suggest_config(trial: object) -> ExperimentConfig:
    """Sample a bounded, explainable search space for this experiment."""
    return ExperimentConfig(
        classifier_hidden_dim=trial.suggest_categorical(
            "classifier_hidden_dim",
            [448, 512, 640],
        ),
        dropout=trial.suggest_float("dropout", 0.02, 0.10, step=0.02),
        layer_mix_depth=trial.suggest_categorical("layer_mix_depth", [3, 4]),
        enable_bitfit=trial.suggest_categorical("enable_bitfit", [True, False]),
        enable_layer_norm_tuning=trial.suggest_categorical(
            "enable_layer_norm_tuning",
            [True, False],
        ),
        lora_rank=trial.suggest_categorical("lora_rank", [12, 16, 24]),
        lora_alpha=trial.suggest_categorical("lora_alpha", [24.0, 32.0, 48.0]),
        lora_dropout=trial.suggest_categorical("lora_dropout", [0.0, 0.05]),
        lora_target_layers=trial.suggest_categorical("lora_target_layers", [8, 12]),
        lora_output_target_layers=trial.suggest_categorical(
            "lora_output_target_layers",
            [2, 4, 6],
        ),
        batch_size=trial.suggest_categorical("batch_size", [12, 16]),
        learning_rate=trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True),
        encoder_learning_rate=trial.suggest_float(
            "encoder_learning_rate",
            1e-5,
            4e-5,
            log=True,
        ),
        weight_decay=trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True),
        epochs=trial.suggest_categorical("epochs", [4]),
        seed=DEFAULT_CONFIG.seed,
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.05, 0.15),
        final_lr_scale=trial.suggest_categorical("final_lr_scale", [0.05, 0.1, 0.2]),
    )


def config_from_trial_params(params: dict[str, object]) -> ExperimentConfig:
    """Convert Optuna's best_params into a concrete experiment config."""
    return ExperimentConfig(
        classifier_hidden_dim=int(params["classifier_hidden_dim"]),
        dropout=float(params["dropout"]),
        layer_mix_depth=int(params["layer_mix_depth"]),
        enable_bitfit=bool(params["enable_bitfit"]),
        enable_layer_norm_tuning=bool(params["enable_layer_norm_tuning"]),
        lora_rank=int(params["lora_rank"]),
        lora_alpha=float(params["lora_alpha"]),
        lora_dropout=float(params["lora_dropout"]),
        lora_target_layers=int(params["lora_target_layers"]),
        lora_output_target_layers=int(params["lora_output_target_layers"]),
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        encoder_learning_rate=float(params["encoder_learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        epochs=int(params["epochs"]),
        seed=DEFAULT_CONFIG.seed,
        warmup_ratio=float(params["warmup_ratio"]),
        final_lr_scale=float(params["final_lr_scale"]),
    )


def export_best_trial(value: float, config: ExperimentConfig) -> None:
    """Persist the best Optuna configuration and score."""
    payload = {
        "value": value,
        "config": asdict(config),
    }
    OPTUNA_BEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def prune_trial(message: str) -> Exception:
    """Create an Optuna pruning exception lazily."""
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Optuna is required for pruning.") from error
    return optuna.TrialPruned(message)


def run_optuna_search(
    n_trials: int,
    timeout_seconds: int | None,
    study_name: str,
) -> None:
    """Run an Optuna TPE search over bounded model and optimizer settings."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna study mode requires the optuna package."
        ) from error

    storage = f"sqlite:///{OPTUNA_STORAGE_PATH.resolve()}"
    sampler = optuna.samplers.TPESampler(seed=DEFAULT_CONFIG.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4,
        n_warmup_steps=2,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        config = suggest_config(trial)
        LOGGER.info("---")
        LOGGER.info("trial=%s config=%s", trial.number, json.dumps(asdict(config)))
        checkpoint_path = CHECKPOINT_DIR / f"trial_{trial.number:04d}_best.pt"
        try:
            summary = run_training(
                config=config,
                checkpoint_path=checkpoint_path,
                save_best=False,
                trial=trial,
            )
        except optuna.TrialPruned:
            append_optuna_result(
                trial_number=trial.number,
                value=float("nan"),
                config=config,
                state="pruned",
            )
            raise
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" in message:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                append_optuna_result(
                    trial_number=trial.number,
                    value=float("nan"),
                    config=config,
                    state="oom",
                )
                raise optuna.TrialPruned("CUDA OOM") from error
            append_optuna_result(
                trial_number=trial.number,
                value=float("nan"),
                config=config,
                state="failed",
            )
            raise

        append_optuna_result(
            trial_number=trial.number,
            value=summary["macro_f1"],
            config=config,
            state="completed",
        )
        trial.set_user_attr("accuracy", summary["accuracy"])
        trial.set_user_attr("val_loss", summary["loss"])
        return summary["macro_f1"]

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    if study.best_trial is None:
        raise RuntimeError("Optuna finished without a best trial.")

    best_config = config_from_trial_params(study.best_trial.params)
    export_best_trial(value=study.best_value, config=best_config)
    LOGGER.info("=== Optuna Best Trial ===")
    LOGGER.info("best_value: %.6f", study.best_value)
    LOGGER.info("best_params: %s", json.dumps(study.best_trial.params, sort_keys=True))
    LOGGER.info("Retraining best config to refresh best_model.pt")
    run_training(
        config=best_config,
        checkpoint_path=CHECKPOINT_PATH,
        save_best=True,
        trial=None,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-run and Optuna study modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        action="store_true",
        help="Run an Optuna parameter search instead of one fixed training run.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
        help="Number of Optuna trials to run when --study is enabled.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Optional wall-clock timeout for the Optuna study.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="project1-bert-advanced",
        help="Stable Optuna study name for local sqlite storage.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    if args.study:
        run_optuna_search(
            n_trials=args.trials,
            timeout_seconds=args.timeout_seconds,
            study_name=args.study_name,
        )
        return
    run_training(
        config=DEFAULT_CONFIG,
        checkpoint_path=CHECKPOINT_PATH,
        save_best=True,
        trial=None,
    )


if __name__ == "__main__":
    main()
