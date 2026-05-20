from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.nolt import NOLTConfig

class NOLTNoCrossSection(nn.Module):
    def __init__(self, config: NOLTConfig):
        super().__init__()
        self.config = config
        flat_token_dim = config.lookback * config.n_features

        self.embed = nn.Linear(flat_token_dim, config.d_model)
        self.embed_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        ffn_hidden = config.d_model * config.dim_feedforward_mult
        layers = []
        for _ in range(config.n_layers):
            layers.append(nn.Sequential(
                nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
                nn.Linear(config.d_model, ffn_hidden),
                nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(ffn_hidden, config.d_model),
                nn.Dropout(config.dropout),
            ))
        self.per_option_layers = nn.ModuleList(layers)
        self.encoder_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

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
        b, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, N, T * F)
        h = self.embed(x)
        h = self.embed_norm(h)

        for layer in self.per_option_layers:
            h = h + layer(h)
        h = self.encoder_norm(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)

class NOLTNoSequence(nn.Module):
    def __init__(self, config: NOLTConfig):
        super().__init__()
        self.config = config

        token_dim = config.n_features

        self.embed = nn.Linear(token_dim, config.d_model)
        self.embed_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_model * config.dim_feedforward_mult,
            dropout=config.dropout, batch_first=True,
            activation=config.activation, norm_first=True,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.encoder_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        b, T, N, F = x.shape

        x_last = x[:, -1, :, :]
        h = self.embed(x_last)
        h = self.embed_norm(h)
        h = self.encoder(h)
        h = self.encoder_norm(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)

class NOLTLinear(nn.Module):
    def __init__(self, config: NOLTConfig):
        super().__init__()
        self.config = config
        in_dim = config.lookback * config.n_options * config.n_features
        self.head = nn.Sequential(
            nn.Linear(in_dim, config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        return self.head(x_flat)
