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


KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class GPTOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None
    past_key_values: KVCache | None = None


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

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, embd_dim = x.shape
        qkv = self.c_attn(x)
        query, key, value = qkv.split(embd_dim, dim=2)

        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_len = past_key.size(-2)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        total_len = past_len + seq_len
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, past_len:total_len, :total_len]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_dim)
        present_key_value = (key, value) if use_cache else None
        return self.resid_dropout(self.c_proj(output)), present_key_value


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

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_output, present_key_value = self.attn(
            self.ln_1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value


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

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> GPTOutput:
        batch_size, seq_len = input_ids.shape
        past_len = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("past_key_values must have one entry per transformer block.")
            past_len = past_key_values[0][0].size(-2)

        total_len = past_len + seq_len
        if total_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence length {total_len}; block_size={self.config.block_size}."
            )
        if labels is not None and use_cache:
            raise ValueError(
                "KV cache is only supported for inference; omit labels when use_cache=True."
            )
        if past_key_values is not None and position_offset == 0:
            position_offset = past_len
        if position_offset < 0 or position_offset + seq_len > self.config.block_size:
            raise ValueError("position_offset places input_ids outside the configured block_size.")

        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            dtype=torch.long,
            device=input_ids.device,
        )
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        present_key_values = []
        for block_idx, block in enumerate(self.blocks):
            past_key_value = None if past_key_values is None else past_key_values[block_idx]
            x, present_key_value = block(
                x,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                if present_key_value is None:
                    raise RuntimeError("Expected a present key/value pair when use_cache=True.")
                present_key_values.append(present_key_value)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, self.config.vocab_size),
                labels.reshape(batch_size * seq_len),
                ignore_index=-100,
            )
        return GPTOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(present_key_values) if use_cache else None,
        )

    def crop_block_size(self, block_size: int) -> None:
        if block_size > self.config.block_size:
            raise ValueError("New block_size must be <= the current block_size.")
        self.position_embedding.weight = nn.Parameter(self.position_embedding.weight[:block_size])
        for block in self.blocks:
            block.attn.causal_mask = block.attn.causal_mask[:, :, :block_size, :block_size]
