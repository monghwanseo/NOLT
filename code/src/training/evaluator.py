import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def auroc(y_true, p) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, p))

def brier(y_true, p) -> float:
    return float(brier_score_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7)))

def ece(y_true, p, n_bins: int = 10) -> float:
    p = np.clip(np.asarray(p), 0, 1)
    y = np.asarray(y_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(p)
    for i in range(n_bins):
        msk = (p >= bin_edges[i]) & (p < bin_edges[i + 1] if i < n_bins - 1
                                      else p <= bin_edges[i + 1])
        if msk.sum() == 0:
            continue
        bin_acc = float(y[msk].mean())
        bin_conf = float(p[msk].mean())
        e += (msk.sum() / n) * abs(bin_acc - bin_conf)
    return float(e)

def evaluate(y_true, p_pred) -> dict:
    return {
        'auroc': auroc(y_true, p_pred),
        'brier': brier(y_true, p_pred),
        'ece': ece(y_true, p_pred, n_bins=10),
        'pos_rate': float(np.mean(y_true)),
        'mean_pred': float(np.mean(p_pred)),
        'n': int(len(y_true)),
    }

def per_period_auroc(dates: pd.DatetimeIndex, y_true, p_pred,
                      freq: str = 'M') -> pd.DataFrame:
    df = pd.DataFrame({'date': dates, 'y': y_true, 'p': p_pred})
    df['period'] = df['date'].dt.to_period(freq)
    rows = []
    for period, g in df.groupby('period'):
        if g['y'].nunique() < 2:
            rows.append({'period': str(period), 'n': len(g),
                         'pos_rate': float(g['y'].mean()), 'auroc': np.nan})
            continue
        rows.append({'period': str(period), 'n': len(g),
                     'pos_rate': float(g['y'].mean()),
                     'auroc': float(roc_auc_score(g['y'], g['p']))})
    return pd.DataFrame(rows)
