# Basic Task Experiment Archive

This note archives the three completed TinyStories GPT size-comparison runs on the HPC.
It follows Sections 2-5 of the Project 2 report requirements and intentionally omits
the Introduction, Discussion, Conclusion, and Advanced Task sections.

## 2. Dataset & Preprocessing

The experiments use TinyStories through Hugging Face `datasets`, with no additional
text data. The training subset is capped at 100,000 TinyStories training examples and
the validation subset is capped at 5,000 validation examples.

Tokenization uses `roneneldan/TinyStories-33M`, a GPT-style BPE tokenizer with a
50,257-token vocabulary. Text is tokenized, concatenated with EOS separators, and
packed into fixed-length autoregressive blocks. Each example returns `input_ids` and
next-token `labels` of length 512.

| Split | Source | Raw examples used | Packed blocks | Block size |
|---|---:|---:|---:|---:|
| Train | `roneneldan/TinyStories` train | 100,000 | 42,679 | 512 |
| Validation | `roneneldan/TinyStories` validation | 5,000 | 1,943 | 512 |

## 3. Model Architectures

All models are implemented from scratch in `gpt.py` using PyTorch modules only. The
architecture contains token embeddings, learned positional embeddings, pre-norm
Transformer decoder blocks, causal multi-head self-attention, MLP feed-forward blocks,
residual connections, a final layer norm, and a tied LM head. No Hugging Face model
classes, `transformers.Trainer`, or `torch.nn.Transformer*` layers are used.

| Model | Layers | Heads | Embedding dim | Context length | Parameters |
|---|---:|---:|---:|---:|---:|
| tiny | 4 | 4 | 256 | 512 | 16.16M |
| small | 8 | 6 | 384 | 512 | 33.69M |
| medium | 12 | 8 | 512 | 512 | 63.82M |

## 4. Training Setup

Training is controlled by Hugging Face Accelerate. The final archived comparison uses
4 visible GPUs with NCCL for all three model sizes. All runs used fp16 mixed precision,
AdamW, cosine learning-rate scheduling with warmup, gradient clipping, TensorBoard
logging, checkpointing every 1,000 steps, and sample generation every 200 validation
steps.

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 8e-4 |
| Adam betas | 0.9, 0.95 |
| Weight decay | 0.1 |
| Warmup ratio | 0.02 |
| Max train steps | 2,000 |
| Per-device batch size | 8 |
| Gradient accumulation | 4 |
| Eval interval | 200 steps |
| Max eval batches | 100 |
| Max grad norm | 1.0 |
| Sampling prompt | `Once upon a time` |
| Sampling | temperature 0.9, top-k 50, top-p 0.95 |

## 5. Results

### Training Curves

The TensorBoard event logs were copied from the HPC into `hpc_logs/` and plotted
with `matplotlib` using `scripts/plot_basic_curves.py`. The `medium` logs are reused
from the earlier 4-GPU run.

![Training and validation loss curves](figures/basic_loss_curves.png)

![Validation perplexity curves](figures/basic_perplexity_curves.png)

### Final Metrics

| Model | Slurm job | GPUs/backend | Train tokens | Runtime | Tokens/sec | Peak allocated memory | Val loss | Val perplexity |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| tiny | 907715 | 4 / NCCL | 131.07M | 7.97 min | 274,182 | 3.79 GiB | 2.270 | 9.68 |
| small | 907707 | 4 / NCCL | 131.07M | 15.20 min | 143,745 | 4.92 GiB | 2.048 | 7.75 |
| medium | 907490 | 4 / NCCL | 131.07M | 26.65 min | 81,969 | 6.64 GiB | 2.108 | 8.23 |

After rerunning `tiny` and `small` on 4 GPUs, all three models use the same total token
budget. The `small` model reaches the best final validation perplexity, while `medium`
has lower final training loss but slightly worse validation loss.

### Validation Curves

| Step | tiny loss | tiny PPL | small loss | small PPL | medium loss | medium PPL |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 3.670 | 39.26 | 3.915 | 50.15 | 4.108 | 60.84 |
| 400 | 3.236 | 25.42 | 3.406 | 30.14 | 3.776 | 43.65 |
| 600 | 3.193 | 24.35 | 3.333 | 28.01 | 3.726 | 41.50 |
| 800 | 2.962 | 19.34 | 2.995 | 19.99 | 3.503 | 33.22 |
| 1000 | 2.662 | 14.33 | 2.559 | 12.92 | 2.990 | 19.88 |
| 1200 | 2.450 | 11.59 | 2.268 | 9.66 | 2.518 | 12.40 |
| 1400 | 2.352 | 10.51 | 2.137 | 8.48 | 2.307 | 10.05 |
| 1600 | 2.344 | 10.42 | 2.127 | 8.39 | 2.288 | 9.85 |
| 1800 | 2.325 | 10.23 | 2.114 | 8.28 | 2.233 | 9.32 |
| 2000 | 2.270 | 9.68 | 2.048 | 7.75 | 2.108 | 8.23 |

### Generated Samples

All samples below use the same prompt: `Once upon a time`. Full text files are stored
under `outputs/<model>/samples/step_2000_sample_*.txt` on the HPC.

| Model | Sample | Excerpt |
|---|---:|---|
| tiny | 1 | A family goes on a trip and arrives at an airport; the setup is coherent but the destination/action drifts. |
| tiny | 2 | Sue plays in the snow and finds a melting snowman; the sample keeps a child-story tone but has awkward logic. |
| tiny | 3 | Lily explores the woods and finds a tree with an axe; the scene is imaginative but object references are confused. |
| small | 1 | Lily goes to the park with her mother, sees swings, slides, and a butterfly; the story is locally coherent. |
| small | 2 | Lily finds a shiny magic chain and reacts to it; the prose is fluent but the chain's role remains unclear. |
| small | 3 | Lily finds a shiny gem in the garden and hears a noise; the story has a clearer event sequence than `tiny`. |
| medium | 1 | Lily sees a butterfly and a cricket in the park; the prose is fluent but the cricket interaction is semantically odd. |
| medium | 2 | Timmy breaks a toy car after spilling it in his room; the story has clearer event progression but imperfect causality. |
| medium | 3 | Lily plays with friends near a swing until it rains; the style matches TinyStories but some phrases remain awkward. |

Overall, increasing from `tiny` to `small` improves validation perplexity clearly under
the same token budget. Increasing further to `medium` does not improve validation
perplexity in this run, suggesting diminishing returns or mild overfitting/optimization
mismatch on the current 100,000-example TinyStories subset.
