from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

def set_deterministic(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-3
    pos_weight_auto: bool = True
    early_stop_patience: int = 15
    grad_clip: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 2026

def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, scores))

def train(model: nn.Module,
          X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray, y_val: np.ndarray,
          cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    set_deterministic(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    if cfg.pos_weight_auto:
        p = float(np.mean(y_train))
        pos_weight = torch.tensor([(1 - p) / max(p, 1e-6)], device=device)
    else:
        pos_weight = None

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().unsqueeze(-1).to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).float().unsqueeze(-1).to(device)

    ds = TensorDataset(Xt, yt)
    g = torch.Generator(); g.manual_seed(cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=g)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_auc = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    train_history, val_history, loss_history = [], [], []
    no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(ds)

        model.eval()
        with torch.no_grad():
            tr_logits = model(Xt).cpu().numpy().squeeze(-1)
            va_logits = model(Xv).cpu().numpy().squeeze(-1)
        tr_auc = _auc(y_train, tr_logits)
        va_auc = _auc(y_val, va_logits)
        train_history.append(tr_auc)
        val_history.append(va_auc)
        loss_history.append(epoch_loss)

        if va_auc > best_val_auc:
            best_val_auc = va_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                break

    model.load_state_dict(best_state)
    return {
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'epochs_run': len(train_history),
        'train_auc_history': train_history,
        'val_auc_history': val_history,
        'loss_history': loss_history,
    }

@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str | None = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    logit = model(X_t).cpu().numpy().squeeze(-1)
    return 1.0 / (1.0 + np.exp(-logit))
