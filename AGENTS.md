# Repository Guidelines

## Project Structure & Module Organization
This repository contains course materials and project scaffolds for CS290S deep learning practice. Top-level lecture slides live in `lectures/` as reference PDFs. Hands-on work belongs in `projects/`, with the current assignment in `projects/project1-text-classification/`.

Within `project1-text-classification/`, extend `data.py` for dataset logic, `train.py` for training and validation loops, and `models/` for architecture implementations such as `your_classifier_model_1.py`. Keep experiment outputs out of source control when possible; use local directories such as `checkpoints/` and `runs/`.

## Build, Test, and Development Commands
Run commands from `projects/project1-text-classification/` unless noted otherwise.

- Environment (required before Python commands): `uv sync && source .venv/bin/activate`
- Linting: `ruff check --fix .` (fixes lint errors)
- Formatting: `ruff format .` (formats code)
- Testing: `python -m pytest` (runs all tests)
- Training entry point: `uv run python train.py`
- Monitoring: `tensorboard --logdir runs`

This project is pinned to Python 3.11 via `.python-version`. Use `uv` as the source of truth for dependency management; keep `requirements.txt` only as a compatibility export when needed. Use shell scripts in `scripts/` to run HPC pipelines; avoid invoking long-running training flows as `python src/run.py` directly.

## Code Style
Act as a careful junior engineer with strong tooling.

- Core: write clean, efficient Python 3.10+. Prefer composition and concise implementations.
- Structure: target 200-400 line files, with 600 as a hard ceiling. Keep functions under 50 lines and organize modules by feature.
- Naming: use `snake_case` for files and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Quality: do not use `print`; use logging. Do not hardcode values; move them to config. Keep nesting to 4 levels max. Use strict type hints and avoid `Any`.
- Best practices: use absolute imports only, write Google-style docstrings, and catch specific exceptions instead of bare `except`.

## Testing Guidelines
There is no dedicated test suite yet, so contributors should validate changes with reproducible training runs. At minimum, verify that `uv run python train.py` starts cleanly, one epoch completes, validation executes, and TensorBoard logs are written.

When adding tests, place them under `projects/project1-text-classification/tests/` and name files `test_*.py`. Prioritize coverage for dataset loading, batch collation, metric calculation, and model forward-pass shapes.

## Commit & Pull Request Guidelines
The current history is minimal (`Initial commit`), so use short, imperative commit messages such as `Implement BiLSTM baseline` or `Add validation F1 logging`. Keep commits focused on one change.

Pull requests should summarize the model or training change, list commands run, report key metrics, and attach plots or screenshots for notable learning-curve differences. Link the related assignment or issue when applicable.
