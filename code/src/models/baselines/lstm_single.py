from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class LSTMConfig:
    n_options: int
    lookback: int
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    n_classes: int = 1

class LSTMSingleOption(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim, cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, T, N), got shape {tuple(x.shape)}")
        b, T, N = x.shape
        if T != self.cfg.lookback or N != self.cfg.n_options:
            raise ValueError(f"input shape mismatch: got T={T}, N={N}, expected "
                             f"T={self.cfg.lookback}, N={self.cfg.n_options}")

        x_perm = x.permute(0, 2, 1).contiguous().view(b * N, T, 1)
        h_seq, (h_last, _) = self.lstm(x_perm)
        h_final = h_seq[:, -1, :]

        logits_per_option = self.head(h_final)
        logits_per_option = logits_per_option.view(b, N, -1)

        return logits_per_option.mean(dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
