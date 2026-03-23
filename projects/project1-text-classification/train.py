"""Reusable training entrypoint for named text classification experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ExperimentConfig, load_experiment_config
from data import (
    TextClassificationDataset,
    TextEncoder,
    build_text_encoder,
    collate_text_classification_batch,
    load_text_examples,
)
from models import build_text_classifier

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("./configs/lstm_classifier.toml")

__all__ = [
    "build_run_name",
    "compute_classification_metrics",
    "create_data_loaders",
    "run_training",
    "validate_model",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment TOML config.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure logging for training runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_name(experiment_name: str) -> str:
    """Build a timestamped TensorBoard run name."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{experiment_name}_{timestamp}"


def create_data_loaders(
    experiment_config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, TextEncoder]:
    """Create train and validation data loaders from config."""
    train_examples = load_text_examples(
        data_config=experiment_config.data,
        csv_path=experiment_config.data.train_csv,
    )
    val_examples = load_text_examples(
        data_config=experiment_config.data,
        csv_path=experiment_config.data.val_csv,
    )
    text_encoder = build_text_encoder(
        data_config=experiment_config.data,
        training_examples=train_examples,
    )

    train_dataset = TextClassificationDataset(
        examples=train_examples,
        text_encoder=text_encoder,
        max_length=experiment_config.data.max_length,
    )
    val_dataset = TextClassificationDataset(
        examples=val_examples,
        text_encoder=text_encoder,
        max_length=experiment_config.data.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=True,
        collate_fn=collate_text_classification_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=False,
        collate_fn=collate_text_classification_batch,
    )
    return train_loader, val_loader, text_encoder


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute binary classification metrics without external metric packages."""
    predictions = logits.argmax(dim=1)

    true_positive = int(((predictions == 1) & (labels == 1)).sum().item())
    true_negative = int(((predictions == 0) & (labels == 0)).sum().item())
    false_positive = int(((predictions == 1) & (labels == 0)).sum().item())
    false_negative = int(((predictions == 0) & (labels == 1)).sum().item())

    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / max(total, 1)

    positive_precision = true_positive / max(true_positive + false_positive, 1)
    positive_recall = true_positive / max(true_positive + false_negative, 1)
    positive_f1 = (
        2
        * positive_precision
        * positive_recall
        / max(positive_precision + positive_recall, 1e-8)
    )

    negative_precision = true_negative / max(true_negative + false_negative, 1)
    negative_recall = true_negative / max(true_negative + false_positive, 1)
    negative_f1 = (
        2
        * negative_precision
        * negative_recall
        / max(negative_precision + negative_recall, 1e-8)
    )

    return {
        "accuracy": accuracy,
        "precision": positive_precision,
        "recall": positive_recall,
        "f1": positive_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
    }


def validate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation for the configured model."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            logits, labels = forward_batch(model=model, batch=batch, device=device)
            loss = F.cross_entropy(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_labels = torch.cat(all_labels, dim=0)
    metrics = compute_classification_metrics(
        logits=stacked_logits,
        labels=stacked_labels,
    )
    metrics["loss"] = total_loss / max(total_examples, 1)

    model.train()
    return metrics


def forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a batch to device and call the model with named inputs."""
    device_batch = {
        feature_name: tensor.to(device) for feature_name, tensor in batch.items()
    }
    labels = device_batch["labels"]
    model_inputs = {
        feature_name: tensor
        for feature_name, tensor in device_batch.items()
        if feature_name != "labels"
    }
    logits = model(**model_inputs)
    return logits, labels


def run_training(experiment_config: ExperimentConfig) -> None:
    """Train the configured model and save the best checkpoint."""
    set_seed(experiment_config.training.seed)
    experiment_config.output.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.output.log_dir.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(experiment_config.experiment_name)
    run_log_dir = experiment_config.output.log_dir / run_name

    train_loader, val_loader, text_encoder = create_data_loaders(experiment_config)
    model = build_text_classifier(
        model_config=experiment_config.model,
        vocabulary_size=text_encoder.vocabulary_size,
        pad_id=text_encoder.pad_id,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=experiment_config.training.learning_rate,
        weight_decay=experiment_config.training.weight_decay,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info(
        (
            "Loaded experiment=%s model=%s train_batches=%s val_batches=%s "
            "device=%s log_dir=%s"
        ),
        experiment_config.experiment_name,
        experiment_config.model.name,
        len(train_loader),
        len(val_loader),
        device,
        run_log_dir,
    )

    writer = SummaryWriter(
        log_dir=str(run_log_dir),
        filename_suffix=f".{run_name}",
    )
    best_macro_f1 = float("-inf")
    total_steps = 0
    checkpoint_path = experiment_config.output.checkpoint_dir / "best_model.pt"

    model.to(device)
    model.train()

    for epoch in range(experiment_config.training.epochs):
        epoch_loss = 0.0
        epoch_examples = 0
        progress_bar = tqdm(
            train_loader,
            desc=(
                f"{experiment_config.model.name} "
                f"Epoch {epoch + 1}/{experiment_config.training.epochs}"
            ),
        )

        for batch in progress_bar:
            optimizer.zero_grad()
            logits, labels = forward_batch(model=model, batch=batch, device=device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_examples += batch_size
            total_steps += 1

            writer.add_scalar("loss/train_step", loss.item(), total_steps)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if total_steps % experiment_config.training.val_steps == 0:
                val_metrics = validate_model(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                )
                log_validation_metrics(
                    writer=writer,
                    metrics=val_metrics,
                    step=total_steps,
                )
                if val_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = val_metrics["macro_f1"]
                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        experiment_config=experiment_config,
                        text_encoder=text_encoder,
                        metrics=val_metrics,
                    )
                    LOGGER.info(
                        "Saved new best checkpoint at step %s with macro F1 %.4f",
                        total_steps,
                        val_metrics["macro_f1"],
                    )

        epoch_train_loss = epoch_loss / max(epoch_examples, 1)
        writer.add_scalar("loss/train_epoch", epoch_train_loss, epoch + 1)

        val_metrics = validate_model(model=model, val_loader=val_loader, device=device)
        log_validation_metrics(writer=writer, metrics=val_metrics, step=total_steps)
        LOGGER.info(
            (
                "Epoch %s complete | train_loss=%.4f val_loss=%.4f "
                "val_acc=%.4f val_f1=%.4f macro_f1=%.4f"
            ),
            epoch + 1,
            epoch_train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["macro_f1"],
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                experiment_config=experiment_config,
                text_encoder=text_encoder,
                metrics=val_metrics,
            )
            LOGGER.info(
                "Saved new best checkpoint after epoch %s with macro F1 %.4f",
                epoch + 1,
                val_metrics["macro_f1"],
            )

    writer.close()


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    experiment_config: ExperimentConfig,
    text_encoder: TextEncoder,
    metrics: dict[str, float],
) -> None:
    """Save a checkpoint with config and encoder metadata."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "experiment_config": experiment_config.to_dict(),
            "text_encoder": text_encoder.export_state(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def log_validation_metrics(
    writer: SummaryWriter,
    metrics: dict[str, float],
    step: int,
) -> None:
    """Write validation metrics to TensorBoard."""
    writer.add_scalar("loss/val", metrics["loss"], step)
    writer.add_scalar("metrics/accuracy", metrics["accuracy"], step)
    writer.add_scalar("metrics/precision", metrics["precision"], step)
    writer.add_scalar("metrics/recall", metrics["recall"], step)
    writer.add_scalar("metrics/f1", metrics["f1"], step)
    writer.add_scalar("metrics/macro_f1", metrics["macro_f1"], step)


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    args = parse_args()
    experiment_config = load_experiment_config(args.config)
    run_training(experiment_config)


if __name__ == "__main__":
    main()
