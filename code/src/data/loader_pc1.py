from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data import config as cfg
from src.data.tasks import common_dates, window_a_tickers, window_b_tickers, window_c_tickers
from src.metrics.bsm_greeks import bsm_call_equivalent_delta, bsm_delta
from src.metrics.linearity import L_B

@dataclass
class LoaderConfig:
    lookback: int = 60
    horizon: int = 1
    threshold_quantile: float = 0.90
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    window: str = 'A'
    seed: int = cfg.SEED

    def __post_init__(self):
        if abs(self.train_frac + self.val_frac + self.test_frac - 1.0) > 1e-9:
            raise ValueError(f"train+val+test = {self.train_frac+self.val_frac+self.test_frac}, expected 1.0")
        if self.window not in ('A', 'B', 'C'):
            raise ValueError(f"window must be 'A', 'B', or 'C'; got {self.window!r}")

def build_residual_matrix(window: str = 'A') -> tuple[pd.DataFrame, list[str]]:
    PROC = cfg.PROCESSED_DIR
    panel = pd.read_parquet(PROC / 'options_panel.parquet')
    qr = pd.read_csv(PROC / 'quality_report.csv', parse_dates=['expiry'])
    spx_pcp = pd.read_parquet(PROC / 'spx_pcp.parquet')
    q_imp = pd.read_parquet(PROC / 'q_implied.parquet')

    if window == 'A':
        ta = window_a_tickers(qr)
    elif window == 'B':
        ta = window_b_tickers(qr)
    elif window == 'C':
        meta = pd.read_parquet(PROC / 'options_meta.parquet')
        ta = window_c_tickers(meta, qr)
    else:
        raise ValueError(f"unknown window: {window!r}")
    cdA = sorted(common_dates(panel, ta))

    sub = panel[panel['ticker'].isin(ta)].dropna(
        subset=['Delta Mid Price', 'Implied Volatility Mid']).copy()
    sub = sub[sub['Date'].dt.date.isin(set(cdA))]
    sub = sub.merge(spx_pcp[['Date', 'S_pcp']], on='Date', how='left')
    sub = sub.merge(q_imp[['Date', 'q_implied']], on='Date', how='left')

    bad_q = (sub['q_implied'] < 0.001) | (sub['q_implied'] > 0.05)
    sub.loc[bad_q, 'q_implied'] = np.nan
    sub['q_used'] = sub['q_implied'].fillna(cfg.Q_BASELINE)

    sub['sigma'] = sub['Implied Volatility Mid'] / 100.0
    sub['tau'] = (sub['expiry'] - sub['Date']).dt.days / 365.25
    sub = sub.dropna(subset=['S_pcp', 'tau']).query('tau > 0').reset_index(drop=True)

    sub['delta_eq_mkt'] = bsm_call_equivalent_delta(
        sub['Delta Mid Price'].values, sub['q_used'].values, sub['tau'].values, sub['option_type'].values)
    sub['delta_bsm'] = bsm_delta(
        sub['S_pcp'].values, sub['strike'].values.astype(float),
        cfg.R, sub['q_used'].values, sub['sigma'].values, sub['tau'].values, sub['option_type'].values)
    sub['delta_eq_bsm'] = bsm_call_equivalent_delta(
        sub['delta_bsm'].values, sub['q_used'].values, sub['tau'].values, sub['option_type'].values)

    bad_eq = (sub['delta_eq_mkt'] < -0.001) | (sub['delta_eq_mkt'] > 1.001)
    sub = sub[~bad_eq].reset_index(drop=True)

    sub['LB_mkt'] = L_B(sub['delta_eq_mkt'].values)
    sub['LB_bsm'] = L_B(sub['delta_eq_bsm'].values)
    sub['residual'] = sub['LB_mkt'] - sub['LB_bsm']

    rmat = sub.pivot_table(index='Date', columns='ticker', values='residual', aggfunc='mean').dropna()
    return rmat, list(rmat.columns)

@dataclass
class PC1Bundle:

    dates: pd.DatetimeIndex
    n_options: int

    R: np.ndarray

    pc1: np.ndarray
    dpc1: np.ndarray

    pca: PCA
    pca_sign: int
    threshold: float

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    pc1_window_train: np.ndarray
    pc1_window_val: np.ndarray
    pc1_window_test: np.ndarray

    sample_dates_train: pd.DatetimeIndex
    sample_dates_val: pd.DatetimeIndex
    sample_dates_test: pd.DatetimeIndex

def build_pc1_bundle(loader_cfg: LoaderConfig | None = None) -> PC1Bundle:
    if loader_cfg is None:
        loader_cfg = LoaderConfig()

    R_df, ticker_list = build_residual_matrix()
    R = R_df.values.astype(np.float64)
    dates = R_df.index
    T_total, N = R.shape

    n_train = int(T_total * loader_cfg.train_frac)
    n_val = int(T_total * loader_cfg.val_frac)
    n_test = T_total - n_train - n_val

    R_train = R[:n_train]
    R_val_test = R[n_train:]

    pca = PCA(n_components=min(5, N), random_state=loader_cfg.seed)
    pca.fit(R_train - R_train.mean(axis=0))

    R_centered = R - R_train.mean(axis=0)
    pc_all = pca.transform(R_centered)
    pc1_raw = pc_all[:, 0]

    abs_residual_sum = np.abs(R).sum(axis=1)
    if np.corrcoef(pc1_raw, abs_residual_sum)[0, 1] < 0:
        pca_sign = -1
    else:
        pca_sign = +1
    pc1 = pc1_raw * pca_sign

    dpc1 = np.diff(pc1)

    dpc1_train = dpc1[:n_train - 1] if n_train >= 2 else dpc1[:0]
    if len(dpc1_train) == 0:
        raise ValueError("Train portion has < 2 samples; cannot compute threshold.")
    threshold = float(np.quantile(np.abs(dpc1_train), loader_cfg.threshold_quantile))

    h = loader_cfg.horizon
    abs_dpc1 = np.abs(dpc1)

    T_lookback = loader_cfg.lookback
    valid_starts = []
    for t in range(T_lookback - 1, T_total - h):
        valid_starts.append(t)
    valid_starts = np.array(valid_starts)

    X_full = np.stack([R[t - T_lookback + 1: t + 1, :] for t in valid_starts], axis=0)
    pc1_window_full = np.stack([pc1[t - T_lookback + 1: t + 1] for t in valid_starts], axis=0)
    label_idx = valid_starts + (h - 1)

    assert (label_idx >= 0).all() and (label_idx < len(abs_dpc1)).all()
    y_full = (abs_dpc1[label_idx] > threshold).astype(np.float32)

    sample_dates = dates[valid_starts]
    is_train = valid_starts < n_train
    is_val = (valid_starts >= n_train) & (valid_starts < n_train + n_val)
    is_test = valid_starts >= n_train + n_val

    return PC1Bundle(
        dates=dates,
        n_options=N,
        R=R,
        pc1=pc1,
        dpc1=dpc1,
        pca=pca,
        pca_sign=pca_sign,
        threshold=threshold,
        X_train=X_full[is_train].astype(np.float32),
        X_val=X_full[is_val].astype(np.float32),
        X_test=X_full[is_test].astype(np.float32),
        y_train=y_full[is_train],
        y_val=y_full[is_val],
        y_test=y_full[is_test],
        pc1_window_train=pc1_window_full[is_train].astype(np.float32),
        pc1_window_val=pc1_window_full[is_val].astype(np.float32),
        pc1_window_test=pc1_window_full[is_test].astype(np.float32),
        sample_dates_train=sample_dates[is_train],
        sample_dates_val=sample_dates[is_val],
        sample_dates_test=sample_dates[is_test],
    )

def build_pc1_bundle_for_fold(train_end: int, val_end: int, test_end: int,
                                lookback: int = 60, horizon: int = 1,
                                threshold_quantile: float = 0.90,
                                seed: int = cfg.SEED,
                                window: str = 'A') -> PC1Bundle:
    R_df, ticker_list = build_residual_matrix(window=window)
    R = R_df.values.astype(np.float64)
    dates = R_df.index
    T_total, N = R.shape

    if not (0 < train_end < val_end < test_end <= T_total):
        raise ValueError(f"need 0 < train_end < val_end < test_end <= {T_total}; got "
                         f"{train_end}, {val_end}, {test_end}")

    R_train = R[:train_end]
    pca = PCA(n_components=min(5, N), random_state=seed)
    pca.fit(R_train - R_train.mean(axis=0))

    R_centered = R - R_train.mean(axis=0)
    pc_all = pca.transform(R_centered)
    pc1_raw = pc_all[:, 0]

    abs_R_sum_train = np.abs(R[:train_end]).sum(axis=1)
    if np.corrcoef(pc1_raw[:train_end], abs_R_sum_train)[0, 1] < 0:
        pca_sign = -1
    else:
        pca_sign = +1
    pc1 = pc1_raw * pca_sign

    dpc1 = np.diff(pc1)
    dpc1_train = dpc1[: train_end - 1] if train_end >= 2 else dpc1[:0]
    if len(dpc1_train) == 0:
        raise ValueError("train portion has < 2 samples")
    threshold = float(np.quantile(np.abs(dpc1_train), threshold_quantile))

    abs_dpc1 = np.abs(dpc1)
    h = horizon
    valid_starts = np.array([t for t in range(lookback - 1, T_total - h)])

    X_full = np.stack([R[t - lookback + 1: t + 1, :] for t in valid_starts], axis=0)
    pc1_window_full = np.stack([pc1[t - lookback + 1: t + 1] for t in valid_starts], axis=0)
    label_idx = valid_starts + (h - 1)
    y_full = (abs_dpc1[label_idx] > threshold).astype(np.float32)

    sample_dates = dates[valid_starts]
    is_train = valid_starts < train_end
    is_val = (valid_starts >= train_end) & (valid_starts < val_end)
    is_test = (valid_starts >= val_end) & (valid_starts < test_end)

    return PC1Bundle(
        dates=dates,
        n_options=N,
        R=R,
        pc1=pc1,
        dpc1=dpc1,
        pca=pca,
        pca_sign=pca_sign,
        threshold=threshold,
        X_train=X_full[is_train].astype(np.float32),
        X_val=X_full[is_val].astype(np.float32),
        X_test=X_full[is_test].astype(np.float32),
        y_train=y_full[is_train],
        y_val=y_full[is_val],
        y_test=y_full[is_test],
        pc1_window_train=pc1_window_full[is_train].astype(np.float32),
        pc1_window_val=pc1_window_full[is_val].astype(np.float32),
        pc1_window_test=pc1_window_full[is_test].astype(np.float32),
        sample_dates_train=sample_dates[is_train],
        sample_dates_val=sample_dates[is_val],
        sample_dates_test=sample_dates[is_test],
    )

def summary(b: PC1Bundle) -> str:
    return (
        f"PC1Bundle: T_total={len(b.dates)}, N={b.n_options}, threshold={b.threshold:.4f}\n"
        f"  pc1 sign aligned: {b.pca_sign:+d}\n"
        f"  PC1 var explained: {b.pca.explained_variance_ratio_[0]:.3f}\n"
        f"  X_train: {b.X_train.shape}  y_train.mean (positive class) = {b.y_train.mean():.3f}\n"
        f"  X_val  : {b.X_val.shape}    y_val.mean   (positive class) = {b.y_val.mean():.3f}\n"
        f"  X_test : {b.X_test.shape}   y_test.mean  (positive class) = {b.y_test.mean():.3f}\n"
        f"  date ranges: train {b.sample_dates_train[0].date()} -> {b.sample_dates_train[-1].date()},  "
        f"val {b.sample_dates_val[0].date()} -> {b.sample_dates_val[-1].date()},  "
        f"test {b.sample_dates_test[0].date()} -> {b.sample_dates_test[-1].date()}"
    )
