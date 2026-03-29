# autoresearch: Project 1 advanced BERT experiment

This repo is now a strict `autoresearch`-style sandbox for one experiment:

- Dataset: `projects/project1-text-classification/data/train.csv` and `val.csv`
- Task: binary sentiment classification
- Backbone: `hfl/chinese-roberta-wwm-ext`
- Baseline architecture: the Project 1 advanced BERT classifier
- Goal: improve validation `macro_f1`

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag, for example `project1-bert-adv-mar29`.
2. Create a branch: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. Run setup verification:
   - `uv sync`
   - `uv run prepare.py`
5. Initialize `results.tsv` with this header if it does not exist:

```tsv
commit	macro_f1	accuracy	memory_gb	status	description
```

6. Confirm setup looks good.

Once setup is confirmed, begin the experiment loop.

## Experimentation

Each experiment is a single run of:

```bash
uv run train.py
```

## Editable scope

You may modify only `train.py`.

That file owns:

- model architecture
- LoRA placement
- classifier head
- dropout
- hidden dimensions
- training hyperparameters
- checkpoint format

## Fixed scope

Do not modify:

- `prepare.py`
- `pyproject.toml`
- `README.md`

`prepare.py` is the fixed data and evaluation harness for this experiment.

## Goal

Optimize in this order:

1. higher `macro_f1`
2. higher `accuracy`
3. lower `val_loss`
4. lower memory and simpler code as tie-breakers

## The first run

The first run should always be the baseline as the file currently stands.

## Output format

The script prints a summary block like:

```text
---
macro_f1: 0.896500
accuracy: 0.945000
val_loss: 0.181000
peak_vram_mb: 4230.1
trainable_params_M: 1.238
```

Extract the result with:

```bash
grep "^macro_f1:\|^accuracy:\|^val_loss:\|^peak_vram_mb:" run.log
```

## Logging results

Log every run to `results.tsv` with five columns:

```tsv
commit	macro_f1	accuracy	memory_gb	status	description
```

Rules:

- use `0.000000` and `0.0` for crashes
- `status` is `keep`, `discard`, or `crash`
- description should be short and specific

Example:

```tsv
commit	macro_f1	accuracy	memory_gb	status	description
a1b2c3d	0.886400	0.940100	4.2	keep	baseline fused pooling plus lora
b2c3d4e	0.889900	0.942000	4.3	keep	increase lora target layers to 6
c3d4e5f	0.884000	0.938700	4.1	discard	switch fused pooling to cls only
```

## Experiment loop

Loop forever:

1. Check git state.
2. Make one focused change in `train.py`.
3. Commit it.
4. Run `uv run train.py > run.log 2>&1`.
5. Read the result summary from `run.log`.
6. If the run crashes, inspect the traceback with `tail -n 50 run.log`.
7. Log the result in `results.tsv`.
8. Keep only meaningful improvements.
9. Revert discarded runs.

## Coursework constraint

This sandbox is for searching over the Project 1 advanced BERT design.

When you are specifically preparing the final "advanced task" comparison for the course report:

- keep the training data fixed
- keep the backbone fixed
- keep the optimizer schedule fixed unless the user explicitly wants hyperparameter search
- prefer architecture changes over pure hyperparameter tuning

## End state

Once a clear winner is found, copy the final architecture and settings from `train.py` back into:

- `projects/project1-text-classification/models/bert_classifier_advanced.py`
- `projects/project1-text-classification/configs/bert_classifier_advanced.toml`

The `autoresearch` repo is the search sandbox. The Project 1 directory remains the final deliverable.
