from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

@dataclass
class NOLTConfig:

    n_options: int
    lookback: int
    n_features: int = 1

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward_mult: int = 4
    dropout: float = 0.1
    activation: str = 'gelu'

    aggregator: str = 'mean'

    n_classes: int = 1

    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.aggregator not in ('mean', 'cls', 'attention_pool'):
            raise ValueError(f"unknown aggregator: {self.aggregator}")
        if self.activation not in ('gelu', 'relu'):
            raise ValueError(f"unknown activation: {self.activation}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

class _AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.query)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.squeeze(1)

class NOLT(nn.Module):

    def __init__(self, config: NOLTConfig):
        super().__init__()
        self.config = config
        flat_token_dim = config.lookback * config.n_features

        self.embed = nn.Linear(flat_token_dim, config.d_model)
        self.embed_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        if config.aggregator == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.dim_feedforward_mult,
            dropout=config.dropout,
            batch_first=True,
            activation=config.activation,
            norm_first=True,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.encoder_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        if config.aggregator == 'attention_pool':
            self.pool = _AttentionPool(config.d_model)
        else:
            self.pool = None

        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(f"expected 3D or 4D input, got shape {tuple(x.shape)}")

        b, T, N, F = x.shape
        if T != cfg.lookback or N != cfg.n_options or F != cfg.n_features:
            raise ValueError(
                f"input shape (B, T, N, F) = {(b, T, N, F)} does not match "
                f"config (T={cfg.lookback}, N={cfg.n_options}, F={cfg.n_features})"
            )

        x = x.permute(0, 2, 1, 3).contiguous().view(b, N, T * F)

        h = self.embed(x)
        h = self.embed_norm(h)

        if self.cls_token is not None:
            cls = self.cls_token.expand(b, -1, -1)
            h = torch.cat([cls, h], dim=1)

        h = self.encoder(h)
        h = self.encoder_norm(h)

        if cfg.aggregator == 'cls':
            pooled = h[:, 0, :]
        elif cfg.aggregator == 'mean':
            pooled = h.mean(dim=1)
        elif cfg.aggregator == 'attention_pool':
            pooled = self.pool(h)
        else:
            raise ValueError(f"unknown aggregator: {cfg.aggregator}")

        return self.head(pooled)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
