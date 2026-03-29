"""Fixed data and evaluation utilities for Project 1 autoresearch runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT1_DIR = PROJECT_ROOT / "projects" / "project1-text-classification"
TRAIN_CSV = PROJECT1_DIR / "data" / "train.csv"
VAL_CSV = PROJECT1_DIR / "data" / "val.csv"
TEXT_FIELD = "review"
LABEL_FIELD = "label"
DEFAULT_PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
DEFAULT_MAX_LENGTH = 256
DEFAULT_NUM_CLASSES = 2

__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_PRETRAINED_MODEL_NAME",
    "TextClassificationDataset",
    "TextExample",
    "TransformerTextEncoder",
    "compute_classification_metrics",
    "evaluate_model",
    "make_dataloaders",
    "verify_setup",
]


@dataclass(frozen=True)
class TextExample:
    """One labeled review from the Project 1 CSV files."""

    text: str
    label: int


class TransformerTextEncoder:
    """Tokenizer wrapper for BERT-style text classification."""

    def __init__(self, pretrained_model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "TransformerTextEncoder requires the transformers package."
            ) from error

        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def encode(self, text: str, max_length: int) -> dict[str, list[int]]:
        """Encode text into padded transformer inputs."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
        }


class TextClassificationDataset(Dataset):
    """Dataset that emits tokenized tensors and labels."""

    def __init__(
        self,
        examples: list[TextExample],
        text_encoder: TransformerTextEncoder,
        max_length: int,
    ) -> None:
        self._examples = examples
        self._text_encoder = text_encoder
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self._examples[index]
        encoded_features = self._text_encoder.encode(
            text=example.text,
            max_length=self._max_length,
        )
        item = {
            feature_name: torch.tensor(feature_value, dtype=torch.long)
            for feature_name, feature_value in encoded_features.items()
        }
        item["labels"] = torch.tensor(example.label, dtype=torch.long)
        return item


def configure_logging() -> None:
    """Configure plain-text logging."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_text_examples(csv_path: Path) -> list[TextExample]:
    """Load labeled reviews from a CSV file."""
    examples: list[TextExample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            examples.append(
                TextExample(
                    text=row[TEXT_FIELD],
                    label=int(row[LABEL_FIELD]),
                )
            )
    return examples


def collate_text_classification_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack a list of example dictionaries into one batch."""
    feature_names = batch[0].keys()
    return {
        feature_name: torch.stack(
            [example[feature_name] for example in batch],
            dim=0,
        )
        for feature_name in feature_names
    }


def make_dataloaders(
    batch_size: int,
    pretrained_model_name: str,
    max_length: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation loaders for the Project 1 dataset."""
    text_encoder = TransformerTextEncoder(pretrained_model_name)
    train_examples = load_text_examples(TRAIN_CSV)
    val_examples = load_text_examples(VAL_CSV)

    train_dataset = TextClassificationDataset(
        examples=train_examples,
        text_encoder=text_encoder,
        max_length=max_length,
    )
    val_dataset = TextClassificationDataset(
        examples=val_examples,
        text_encoder=text_encoder,
        max_length=max_length,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_text_classification_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_text_classification_batch,
    )
    return train_loader, val_loader


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute binary-classification metrics without external packages."""
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


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a batch to device and run the model."""
    device_batch = {
        feature_name: tensor.to(device) for feature_name, tensor in batch.items()
    }
    labels = device_batch["labels"]
    logits = model(
        input_ids=device_batch["input_ids"],
        attention_mask=device_batch["attention_mask"],
    )
    return logits, labels


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run fixed validation for autoresearch experiments."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in val_loader:
        logits, labels = _forward_batch(model=model, batch=batch, device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)

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


def verify_setup(pretrained_model_name: str) -> None:
    """Verify that the dataset exists and the tokenizer can be loaded."""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing training data: {TRAIN_CSV}")
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Missing validation data: {VAL_CSV}")

    train_examples = load_text_examples(TRAIN_CSV)
    val_examples = load_text_examples(VAL_CSV)
    TransformerTextEncoder(pretrained_model_name)

    LOGGER.info("Project 1 directory: %s", PROJECT1_DIR)
    LOGGER.info("Training examples: %s", len(train_examples))
    LOGGER.info("Validation examples: %s", len(val_examples))
    LOGGER.info("Tokenizer OK: %s", pretrained_model_name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_PRETRAINED_MODEL_NAME,
        help="Hugging Face model name for tokenizer verification.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for one-time setup verification."""
    configure_logging()
    args = parse_args()
    verify_setup(pretrained_model_name=args.model_name)
    LOGGER.info("Setup verification complete.")


if __name__ == "__main__":
    main()
