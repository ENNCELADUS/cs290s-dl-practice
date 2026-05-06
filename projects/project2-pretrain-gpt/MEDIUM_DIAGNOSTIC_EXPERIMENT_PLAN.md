# Medium Hyperparameter Diagnostic Plan

## Hypothesis

The current `medium` model underperforms `small` on validation perplexity because its
optimization or regularization is not matched to the larger capacity. The main
symptom is lower final training loss than `small` but worse validation loss, so the
diagnostic runs test whether reducing optimizer aggressiveness or increasing
regularization improves validation perplexity.

## Baseline

Use the existing 4-GPU `medium` run as the baseline:

| Config | LR | Warmup | Weight decay | Train tokens | Final val loss | Final PPL |
|---|---:|---:|---:|---:|---:|---:|
| `configs/experiments/medium.yaml` | 8e-4 | 0.02 | 0.1 | 131.07M | 2.108 | 8.23 |

## Controlled Variables

All diagnostic runs keep these variables fixed:

| Variable | Value |
|---|---:|
| Model | `configs/models/medium.json` |
| Dataset | `roneneldan/TinyStories` |
| Train subset | 100,000 examples |
| Validation subset | 5,000 examples |
| Tokenizer | `roneneldan/TinyStories-33M` |
| Context length | 512 |
| Seed | 42 |
| GPUs | 4 |
| Per-device batch size | 8 |
| Gradient accumulation | 4 |
| Max steps | 2,000 |
| Effective train tokens | 131.07M |
| Eval interval | 200 steps |
| Sampling settings | same prompt, temperature, top-k, top-p |

## Diagnostic Conditions

| Config | Changed variable | Reason |
|---|---|---|
| `configs/experiments/diagnostics/medium_lr_6e4.yaml` | LR 8e-4 -> 6e-4 | Tests whether the baseline LR is slightly too aggressive for the larger model. |
| `configs/experiments/diagnostics/medium_lr_4e4.yaml` | LR 8e-4 -> 4e-4 | Tests a stronger LR reduction; useful if validation improves late but training slows. |
| `configs/experiments/diagnostics/medium_warmup_5pct.yaml` | warmup 0.02 -> 0.05 | Tests whether early optimization is too abrupt for `medium`. |
| `configs/experiments/diagnostics/medium_wd_02.yaml` | weight decay 0.1 -> 0.2 | Tests whether stronger regularization closes the train/validation gap. |

## Metrics

Primary metric:
- final validation perplexity at 2,000 steps.

Secondary metrics:
- final validation loss
- validation curve shape from steps 200-2,000
- train loss vs. validation loss gap
- training throughput and peak GPU memory
- generated samples from the same prompt

## Success Criteria

A diagnostic config is promising if it beats the baseline `medium` final perplexity
of 8.23 and ideally matches or beats the current `small` final perplexity of 7.75.

Interpretation:
- Lower LR helps: the baseline was likely over-aggressive for the larger model.
- Longer warmup helps: instability or early optimization dynamics were limiting.
- Higher weight decay helps: the model was mildly overfitting the 100k-example subset.
- None help: the issue is likely data/token budget or architecture size mismatch, not
  these simple training hyperparameters.

## Compute Budget

Each run uses 4 GPUs and should be comparable to the previous `medium` runtime
of about 26-27 minutes. Four diagnostic runs cost roughly 1.8 GPU-hours each,
or about 7.2 GPU-hours total.
