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

Training is controlled by Hugging Face Accelerate. The `tiny` and `small` runs used
2 visible GPUs with the `gloo` distributed backend because the allocated node had
unhealthy GPUs filtered out by the launcher. The `medium` run used 4 visible GPUs with
NCCL. All runs used fp16 mixed precision, AdamW, cosine learning-rate scheduling with
warmup, gradient clipping, TensorBoard logging, checkpointing every 1,000 steps, and
sample generation every 200 validation steps.

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

The TensorBoard event logs were copied from the HPC into `hpc_logs/` and plotted with
`matplotlib` using `scripts/plot_basic_curves.py`.

![Training and validation loss curves](figures/basic_loss_curves.png)

![Validation perplexity curves](figures/basic_perplexity_curves.png)

### Final Metrics

| Model | Slurm job | GPUs/backend | Train tokens | Runtime | Tokens/sec | Peak allocated memory | Val loss | Val perplexity |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| tiny | 907478 | 2 / gloo | 65.54M | 8.94 min | 122,124 | 3.79 GiB | 2.419 | 11.24 |
| small | 907489 | 2 / gloo | 65.54M | 15.95 min | 68,472 | 4.91 GiB | 2.188 | 8.92 |
| medium | 907490 | 4 / NCCL | 131.07M | 26.65 min | 81,969 | 6.64 GiB | 2.108 | 8.23 |

The `medium` run reached the best final validation perplexity, but it processed twice
as many total tokens because it ran with 4 GPUs while `tiny` and `small` ran with 2.
The cleanest equal-token comparison is therefore `tiny` vs. `small`.

### Validation Curves

| Step | tiny loss | tiny PPL | small loss | small PPL | medium loss | medium PPL |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 3.774 | 43.54 | 3.750 | 42.51 | 4.108 | 60.84 |
| 400 | 3.182 | 24.10 | 3.036 | 20.83 | 3.776 | 43.65 |
| 600 | 2.907 | 18.31 | 2.693 | 14.78 | 3.726 | 41.50 |
| 800 | 2.789 | 16.26 | 2.549 | 12.80 | 3.503 | 33.22 |
| 1000 | 2.765 | 15.88 | 2.517 | 12.39 | 2.990 | 19.88 |
| 1200 | 2.756 | 15.74 | 2.506 | 12.25 | 2.518 | 12.40 |
| 1400 | 2.704 | 14.94 | 2.460 | 11.71 | 2.307 | 10.05 |
| 1600 | 2.613 | 13.65 | 2.377 | 10.77 | 2.288 | 9.85 |
| 1800 | 2.516 | 12.38 | 2.278 | 9.76 | 2.233 | 9.32 |
| 2000 | 2.419 | 11.24 | 2.188 | 8.92 | 2.108 | 8.23 |

### Generated Samples

All samples below use the same prompt: `Once upon a time`. Full text files are stored
under `outputs/<model>/samples/step_2000_sample_*.txt` on the HPC.

| Model | Sample | Excerpt |
|---|---:|---|
| tiny | 1 | A little girl named Lily plays outside with her mom, then asks to play hide and seek near the swings. |
| tiny | 2 | Lily repeatedly plays with toys, goes to a store, and finds an old doll; the story restarts partway through. |
| tiny | 3 | Lily sees a big dog in the park, but the dialogue becomes inconsistent and the moral ending is weak. |
| small | 1 | Billy and Timmy appear in a simple family story about going to the post office and buying things. |
| small | 2 | Lily plays in the park, talks with her mother, and begins counting, though some object references are unclear. |
| small | 3 | Lily and her mother discuss buying something; the story is coherent at sentence level but repeats money/buying. |
| medium | 1 | Lily sees a butterfly and a cricket in the park; the prose is fluent but the cricket interaction is semantically odd. |
| medium | 2 | Timmy breaks a toy car after spilling it in his room; the story has clearer event progression but imperfect causality. |
| medium | 3 | Lily plays with friends near a swing until it rains; the style matches TinyStories but some phrases remain awkward. |

Overall, larger models reduce validation perplexity and improve local fluency. The
`small` model gives the strongest equal-token improvement over `tiny`; `medium` is the
best absolute run but should be reported with the token-budget caveat above.
