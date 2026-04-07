# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS290S Deep Learning Practice - a course repository for text classification using Chinese sentiment analysis. The main coursework is in `projects/project1-text-classification/`, with an isolated optimization sandbox in `autoresearch/`.

**Target Performance**: Accuracy > 0.92 or Macro F1 > 0.86

## Development Commands

All commands run from `projects/project1-text-classification/`:

```bash
# Environment setup
uv sync                              # Install dependencies (source of truth)
source .venv/bin/activate            # Activate environment

# Training
uv run python train.py --config ./configs/lstm_classifier.toml
uv run python train.py --config ./configs/bert_classifier.toml
uv run python train.py --config ./configs/bert_classifier_advanced.toml

# Evaluation
uv run python evaluate.py --checkpoint ./checkpoints/best_model.pt --config ./configs/lstm_classifier.toml

# Code quality
ruff check --fix .                   # Lint and auto-fix
ruff format .                        # Format code
python -m pytest                     # Run tests

# Monitoring
tensorboard --logdir ./runs          # View training metrics
```

**Python version**: 3.11 (pinned in `.python-version`)

## Architecture

### Config-Driven Training Pipeline

The codebase uses frozen dataclasses for type-safe configuration:

- **DataConfig**: Dataset paths, encoding type (character/transformer), max_length, vocab settings
- **ModelConfig**: Model selection and architecture hyperparameters
- **TrainingConfig**: Optimizer, LR, epochs, validation frequency, scheduler
- **OutputConfig**: Checkpoint and log directories

Configs are TOML files in `configs/`. The system resolves relative paths from project root.

### Two Text Encoding Strategies

1. **CharacterTextEncoder** (`data.py`):
   - Builds character-level vocabulary from training data
   - Returns `input_ids` and `lengths` for `pack_padded_sequence`
   - Used by LSTM models

2. **TransformerTextEncoder** (`data.py`):
   - Wraps HuggingFace tokenizers (e.g., `hfl/chinese-roberta-wwm-ext`)
   - Returns `input_ids` and `attention_mask`
   - Used by BERT models

### Model Architectures

**Baseline Models** (`models/baseline.py`):
- `LSTMClassifier`: Bidirectional LSTM with mean pooling
- `BertClassifier`: Frozen/fine-tuned BERT with configurable pooling (CLS/mean/max)

**Advanced Models** (`models/advanced.py`):
- `LSTMClassifierAdvanced`: Multi-layer BiLSTM with attention and residual connections
- `BertClassifierAdvanced`: BERT with LoRA adapters, fused pooling, and optional encoder freezing

All models implement `forward(input_ids, **kwargs) -> logits` and `get_loss(logits, labels) -> loss`.

### Training Loop

`train.py` orchestrates:
1. Load config from TOML
2. Initialize encoder, dataset, model
3. Training loop with gradient accumulation support
4. Validation every N steps
5. Checkpoint best model (by val loss)
6. Log metrics to TensorBoard

Key features:
- Automatic mixed precision (AMP) support
- Learning rate scheduling (StepLR, CosineAnnealingLR)
- Early stopping via validation monitoring
- Timestamped TensorBoard runs

### Evaluation

`evaluate.py` computes:
- Accuracy
- Precision, Recall, F1 (macro and per-class)
- Confusion matrix
- Classification report

Loads checkpoint and config, runs inference on validation set.

### Utilities

- `utils/data_processing.py`: CSV loading, train/val split
- `utils/evaluation.py`: Metric computation functions
- `utils/optuna_utils.py`: Hyperparameter optimization with Optuna (used in `autoresearch/`)

## Autoresearch Sandbox

`autoresearch/` is an **isolated experimentation environment** for advanced optimization:

- Separate `pyproject.toml` with additional dependencies (optuna, alembic)
- Database-backed study tracking
- Hyperparameter sweep scripts
- Does NOT affect main project code

Use this for exploring optimization strategies before integrating into main project.

## Code Standards (from AGENTS.md)

- **Module size**: 200-400 lines target, 600 hard max
- **Function size**: Max 50 lines
- **Naming**: snake_case (files/functions), PascalCase (classes), UPPER_SNAKE_CASE (constants)
- **Type hints**: Strict, no `Any`
- **Docstrings**: Google style
- **Imports**: Absolute only
- **No print statements**: Use logging module
- **Max nesting depth**: 4 levels

## Dataset

Location: `projects/project1-text-classification/data/`

- `train.csv`: ~8,000 Chinese text reviews with binary sentiment labels
- `val.csv`: ~2,000 samples
- Format: `review` (text), `label` (0=negative, 1=positive)

## Key Implementation Notes

1. **Character encoding** builds vocabulary dynamically from training data - ensure `CharacterVocabulary` is saved/loaded with model checkpoints
2. **BERT models** require `attention_mask` in forward pass - TransformerTextEncoder provides this
3. **LSTM models** benefit from `pack_padded_sequence` - use `lengths` from CharacterTextEncoder
4. **Advanced models** have many hyperparameters - use config files to manage complexity
5. **Checkpoints** save full model state_dict - load with `torch.load()` and `model.load_state_dict()`
6. **TensorBoard logs** are timestamped - each run creates a new subdirectory in `runs/`

## Common Workflows

### Adding a New Model

1. Implement in `models/baseline.py` or `models/advanced.py`
2. Follow interface: `forward(input_ids, **kwargs) -> logits` and `get_loss(logits, labels) -> loss`
3. Register in `config.py` ModelConfig
4. Create config TOML in `configs/`
5. Test with `train.py`

### Hyperparameter Tuning

Use `autoresearch/` for systematic exploration:
1. Define search space in Optuna script
2. Run study with database backend
3. Analyze results
4. Port best config to main project

### Debugging Training Issues

1. Check TensorBoard for loss curves: `tensorboard --logdir ./runs`
2. Verify data loading: inspect batch shapes and values
3. Test model forward pass in isolation
4. Check learning rate schedule
5. Monitor gradient norms (add logging if needed)
