from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

SEED = 2026

def set_deterministic(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class SyntheticDataset(Dataset):
    R: np.ndarray
    dpc1: np.ndarray
    threshold: float
    lookback: int
    path_indices: np.ndarray
    stride: int = 1

    def __post_init__(self):

        T_total = self.R.shape[1]
        valid_t = np.arange(self.lookback - 1, T_total - 1, self.stride)

        path_idx_grid, t_grid = np.meshgrid(self.path_indices, valid_t, indexing="ij")
        self._path_idx = path_idx_grid.ravel()
        self._t_idx = t_grid.ravel()

    def __len__(self) -> int:
        return len(self._path_idx)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._path_idx[i]
        t = self._t_idx[i]
        window = self.R[p, t - self.lookback + 1 : t + 1, :]
        label = float(abs(self.dpc1[p, t]) > self.threshold)
        return torch.from_numpy(window.astype(np.float32)), torch.tensor([label], dtype=torch.float32)

@dataclass
class PretrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    early_stop_patience: int = 8
    grad_clip: float = 1.0
    pos_weight_auto: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = SEED
    num_workers: int = 0

@dataclass
class PretrainResult:
    best_val_auc: float
    best_epoch: int
    epochs_run: int
    train_auc_history: list
    val_auc_history: list
    loss_history: list
    test_auc: float
    sanity_pass: bool
    sanity_details: dict

def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    labels = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logit = model(xb).cpu().numpy().squeeze(-1)
        scores.append(logit)
        labels.append(yb.numpy().squeeze(-1))
    s = np.concatenate(scores)
    l = np.concatenate(labels)
    return _auc(l, s), s, l

def _check_sanity(train_history: list, val_history: list,
                  train_auc_min: float = 0.55,
                  overfit_gap_max: float = 0.15,
                  val_min_rise_threshold: float = 0.005) -> tuple[bool, dict]:
    if not train_history:
        return False, {"reason": "empty history"}
    best_train = max(train_history)
    best_val = max(val_history)
    val_rise = best_val - val_history[0]
    overfit_gap = best_train - best_val

    train_pass = best_train >= train_auc_min
    val_pass = val_rise >= val_min_rise_threshold
    gap_pass = overfit_gap <= overfit_gap_max
    all_pass = train_pass and val_pass and gap_pass
    details = {
        "best_train_auc": float(best_train),
        "best_val_auc": float(best_val),
        "val_rise": float(val_rise),
        "overfit_gap": float(overfit_gap),
        "train_pass": bool(train_pass),
        "val_pass": bool(val_pass),
        "gap_pass": bool(gap_pass),
    }
    return all_pass, details

def pretrain_nolt(model: nn.Module,
                  train_ds: Dataset,
                  val_ds: Dataset,
                  test_ds: Dataset,
                  cfg: PretrainConfig | None = None) -> PretrainResult:
    cfg = cfg or PretrainConfig()
    set_deterministic(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              generator=g, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    if cfg.pos_weight_auto:

        n_sample = min(5000, len(train_ds))
        pos_count = 0
        for i in range(n_sample):
            _, y = train_ds[i]
            pos_count += int(y.item())
        p = pos_count / n_sample
        pos_weight = torch.tensor([(1 - p) / max(p, 1e-6)], device=device)
    else:
        pos_weight = None

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_auc = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    train_hist, val_hist, loss_hist = [], [], []
    no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            epoch_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        epoch_loss /= max(n_seen, 1)

        train_auc, _, _ = _evaluate(model, train_loader, device)
        val_auc, _, _ = _evaluate(model, val_loader, device)
        train_hist.append(train_auc)
        val_hist.append(val_auc)
        loss_hist.append(epoch_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                break

    model.load_state_dict(best_state)

    test_auc, _, _ = _evaluate(model, test_loader, device)

    sanity_pass, sanity_details = _check_sanity(train_hist, val_hist)

    return PretrainResult(
        best_val_auc=float(best_val_auc),
        best_epoch=int(best_epoch),
        epochs_run=len(train_hist),
        train_auc_history=train_hist,
        val_auc_history=val_hist,
        loss_history=loss_hist,
        test_auc=float(test_auc),
        sanity_pass=bool(sanity_pass),
        sanity_details=sanity_details,
    )

def build_pretrain_datasets(npz_path: str | Path,
                             lookback: int,
                             stride: int = 1,
                             train_frac: float = 0.70,
                             val_frac: float = 0.15,
                             threshold_quantile: float = 0.90,
                             seed: int = SEED) -> tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset, dict]:
    npz_path = Path(npz_path)
    arr = np.load(npz_path)
    R = arr["R"]
    dpc1 = arr["dpc1"]

    n_paths = R.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_paths)
    n_train = int(round(n_paths * train_frac))
    n_val = int(round(n_paths * val_frac))
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val:])

    train_dpc1 = dpc1[train_idx].flatten()
    threshold = float(np.quantile(np.abs(train_dpc1), threshold_quantile))

    train_ds = SyntheticDataset(R=R, dpc1=dpc1, threshold=threshold,
                                lookback=lookback, path_indices=train_idx, stride=stride)
    val_ds = SyntheticDataset(R=R, dpc1=dpc1, threshold=threshold,
                              lookback=lookback, path_indices=val_idx, stride=stride)
    test_ds = SyntheticDataset(R=R, dpc1=dpc1, threshold=threshold,
                               lookback=lookback, path_indices=test_idx, stride=stride)
    info = {
        "n_paths_total": int(n_paths),
        "n_paths_train": int(len(train_idx)),
        "n_paths_val": int(len(val_idx)),
        "n_paths_test": int(len(test_idx)),
        "n_samples_train": len(train_ds),
        "n_samples_val": len(val_ds),
        "n_samples_test": len(test_ds),
        "threshold": threshold,
        "lookback": lookback,
        "stride": stride,
        "seed": seed,
    }
    return train_ds, val_ds, test_ds, info
