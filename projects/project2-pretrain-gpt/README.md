# Project 2: From-Scratch GPT Pretraining

This scaffold trains a handwritten GPT on TinyStories while using Hugging Face
only for tokenizer/data infrastructure and `accelerate` for DDP.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Smoke Test

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml train.py \
  --config configs/experiments/smoke.yaml
```

## Full Size Sweep

```bash
sbatch scripts/run_hpc.sh tiny
sbatch scripts/run_hpc.sh small
sbatch scripts/run_hpc.sh medium
```

The comparable model configs are in `configs/models/`. The experiment configs
keep dataset, tokenizer, sequence length, optimizer, schedule, and token budget
fixed across the basic-task size comparison.

Each full run writes TensorBoard logs, three generated samples under
`outputs/<run>/samples/`, checkpoints under `outputs/<run>/checkpoint-*`, and a
`final_metrics.json` file for the report table.

## Generation

```bash
python generate.py \
  --checkpoint outputs/small/checkpoint-2000/model.pt \
  --model-config configs/models/small.json \
  --prompt "Once upon a time"
```

Do not include TinyStories data, checkpoints, or `outputs/` in the final code zip.
