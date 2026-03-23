"""BiLSTM classifier for character-level text classification."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["LstmClassifier"]


class LstmClassifier(nn.Module):
    """Character-level BiLSTM classifier."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_id: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded_tokens = self.embedding(input_ids)
        packed_tokens = nn.utils.rnn.pack_padded_sequence(
            input=embedded_tokens,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_states, _) = self.encoder(packed_tokens)

        forward_hidden = hidden_states[-2]
        backward_hidden = hidden_states[-1]
        encoded_features = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(self.dropout(encoded_features))
