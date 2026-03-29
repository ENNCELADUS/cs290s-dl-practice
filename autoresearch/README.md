# autoresearch for CS290S Project 1

This is a forked `autoresearch` sandbox for one specific experiment: the advanced BERT classifier from Project 1.

The structure now follows Karpathy's organization closely:

- `prepare.py` is fixed and owns dataset access plus evaluation.
- `train.py` is the only experiment surface and contains the model architecture and hyperparameters.
- `program.md` tells an agent how to run the keep/discard experiment loop.

The training data is reused from:

- `../projects/project1-text-classification/data/train.csv`
- `../projects/project1-text-classification/data/val.csv`

## Quick start

```bash
cd autoresearch
uv sync
uv run prepare.py
uv run train.py
```

## Files that matter

```text
prepare.py      fixed Project 1 data and evaluation utilities
train.py        advanced BERT architecture plus training hyperparameters
program.md      agent instructions for the experiment loop
pyproject.toml  dependencies for this sandbox
```

## How to use it

1. Run the baseline once.
2. Let the agent modify only `train.py`.
3. Compare runs using `macro_f1` first.
4. Keep only meaningful improvements.
5. When you have a winner, copy that architecture and parameter set back into the Project 1 repo.

This keeps the search process isolated while preserving a clean final coursework implementation.
