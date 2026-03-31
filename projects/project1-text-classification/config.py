"""Configuration loading for text classification experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib

ScalarConfigValue = bool | float | int | str

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OutputConfig",
    "TrainingConfig",
    "load_experiment_config",
]


@dataclass(frozen=True)
class DataConfig:
    """Dataset and encoding configuration."""

    train_csv: Path
    val_csv: Path
    text_field: str
    label_field: str
    encoding: str
    max_length: int
    min_frequency: int = 1
    pretrained_model_name: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and model-specific parameters."""

    name: str
    parameters: dict[str, ScalarConfigValue]


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and trainer settings."""

    batch_size: int
    learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    epochs: int
    val_steps: int
    seed: int
    warmup_ratio: float
    final_lr_scale: float


@dataclass(frozen=True)
class OutputConfig:
    """Paths for checkpoints and TensorBoard logs."""

    checkpoint_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Full experiment configuration."""

    experiment_name: str
    config_path: Path
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the config."""
        raw = asdict(self)
        return {
            "experiment_name": raw["experiment_name"],
            "config_path": str(self.config_path),
            "data": _stringify_paths(raw["data"]),
            "model": raw["model"],
            "training": raw["training"],
            "output": _stringify_paths(raw["output"]),
        }


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a TOML file."""
    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    base_dir = resolved_config_path.parent.parent

    data_section = raw_config["data"]
    training_section = raw_config["training"]
    output_section = raw_config["output"]
    model_section = raw_config["model"]

    return ExperimentConfig(
        experiment_name=raw_config["experiment"]["name"],
        config_path=resolved_config_path,
        data=DataConfig(
            train_csv=_resolve_path(base_dir, data_section["train_csv"]),
            val_csv=_resolve_path(base_dir, data_section["val_csv"]),
            text_field=data_section.get("text_field", "review"),
            label_field=data_section.get("label_field", "label"),
            encoding=data_section["encoding"],
            max_length=int(data_section["max_length"]),
            min_frequency=int(data_section.get("min_frequency", 1)),
            pretrained_model_name=data_section.get("pretrained_model_name"),
        ),
        model=ModelConfig(
            name=model_section["name"],
            parameters={
                key: value
                for key, value in raw_config.get("model_parameters", {}).items()
            },
        ),
        training=TrainingConfig(
            batch_size=int(training_section["batch_size"]),
            learning_rate=float(training_section["learning_rate"]),
            encoder_learning_rate=float(
                training_section.get(
                    "encoder_learning_rate",
                    training_section["learning_rate"],
                )
            ),
            weight_decay=float(training_section["weight_decay"]),
            epochs=int(training_section["epochs"]),
            val_steps=int(training_section["val_steps"]),
            seed=int(training_section["seed"]),
            warmup_ratio=float(training_section.get("warmup_ratio", 0.0)),
            final_lr_scale=float(training_section.get("final_lr_scale", 1.0)),
        ),
        output=OutputConfig(
            checkpoint_dir=_resolve_path(base_dir, output_section["checkpoint_dir"]),
            log_dir=_resolve_path(base_dir, output_section["log_dir"]),
        ),
    )


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    """Resolve relative config paths against the project root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _stringify_paths(raw_dict: dict[str, object]) -> dict[str, object]:
    """Convert Path values to strings recursively."""
    stringified: dict[str, object] = {}
    for key, value in raw_dict.items():
        if isinstance(value, Path):
            stringified[key] = str(value)
        else:
            stringified[key] = value
    return stringified
