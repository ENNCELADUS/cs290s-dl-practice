from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.vocab_size <= 0 or self.block_size <= 0:
            raise ValueError("vocab_size and block_size must be positive.")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class GPTOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embd_dim = x.shape
        qkv = self.c_attn(x)
        query, key, value = qkv.split(embd_dim, dim=2)

        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_dim)
        return self.resid_dropout(self.c_proj(output))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> GPTOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence length {seq_len}; block_size={self.config.block_size}."
            )

        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, self.config.vocab_size),
                labels.reshape(batch_size * seq_len),
                ignore_index=-100,
            )
        return GPTOutput(logits=logits, loss=loss)

    def crop_block_size(self, block_size: int) -> None:
        if block_size > self.config.block_size:
            raise ValueError("New block_size must be <= the current block_size.")
        self.position_embedding.weight = nn.Parameter(self.position_embedding.weight[:block_size])
        for block in self.blocks:
            block.attn.causal_mask = block.attn.causal_mask[:, :, :block_size, :block_size]
