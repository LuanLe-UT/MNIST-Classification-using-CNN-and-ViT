"""Manually defined ViT layers; PyTorch's built-in attention is used."""

import math

import torch
from torch import nn


class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(input_size, output_size) * math.sqrt(2.0 / input_size)
        )
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        return x @ self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        return normalized * self.scale + self.shift


class Dropout(nn.Module):
    def __init__(self, probability=0.1):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("probability must be in the range [0, 1)")
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        mask = (torch.rand_like(x) >= self.probability).to(x.dtype)
        return x * mask / (1 - self.probability)


class PositionalEncoding(nn.Module):
    """Create sinusoidal positions without a built-in encoding layer."""

    def __init__(self, sequence_length, embedding_dim):
        super().__init__()
        positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(embedding_dim, dtype=torch.float32).unsqueeze(0)
        angles = positions / torch.pow(
            torch.tensor(10000.0), (2 * torch.floor(dimensions / 2)) / embedding_dim
        )
        encoding = torch.empty(sequence_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        encoding[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, tokens):
        if tokens.shape[1] > self.encoding.shape[1]:
            raise ValueError("token sequence exceeds positional encoding length")
        return tokens + self.encoding[:, :tokens.shape[1]].to(dtype=tokens.dtype)


class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.expand = Dense(embedding_dim, hidden_dim)
        self.dropout1 = Dropout(dropout)
        self.contract = Dense(hidden_dim, embedding_dim)
        self.dropout2 = Dropout(dropout)

    def forward(self, x):
        x = self.expand(x)
        x = 0.5 * x * (1 + torch.erf(x / math.sqrt(2.0)))
        x = self.dropout1(x)
        return self.dropout2(self.contract(x))


class EncoderBlock(nn.Module):
    """Pre-norm encoder: Norm -> MHSA -> residual -> Norm -> MLP -> residual."""

    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention_norm = LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = Dropout(dropout)
        self.mlp_norm = LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, dropout)

    def forward(self, x):
        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        x = x + self.attention_dropout(attended)
        return x + self.mlp(self.mlp_norm(x))
