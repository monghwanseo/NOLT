import numpy as np
import pandas as pd

def L_B(call_eq_delta) -> np.ndarray:
    d = np.asarray(call_eq_delta, dtype=float)
    return (2.0 * d - 1.0) ** 2

def L_A(gamma_normalized) -> np.ndarray:
    g = np.asarray(gamma_normalized, dtype=float)
    g = np.clip(g, 1e-12, 1.0 / np.sqrt(2 * np.pi))

    raw = -np.log(g) + np.log(1.0 / np.sqrt(2 * np.pi))

    return 1.0 - np.exp(-raw)

def L_C_rolling(price_change: pd.Series, delta_times_dS: pd.Series, window: int = 20) -> pd.Series:
    df = pd.DataFrame({'dP': price_change, 'dHat': delta_times_dS}).dropna()
    if len(df) < window:
        return pd.Series(index=price_change.index, dtype=float)

    out = pd.Series(index=df.index, dtype=float)
    for i in range(window - 1, len(df)):
        sl = df.iloc[i - window + 1: i + 1]
        if sl['dHat'].std() == 0 or sl['dP'].std() == 0:
            out.iloc[i] = np.nan
            continue
        corr = np.corrcoef(sl['dP'], sl['dHat'])[0, 1]
        out.iloc[i] = corr * corr
    return out.reindex(price_change.index)

def L_D(price_change, delta, gamma, dS, dt, theta=None) -> np.ndarray:
    dP = np.asarray(price_change, dtype=float)
    Delta = np.asarray(delta, dtype=float)
    G = np.asarray(gamma, dtype=float)
    dS = np.asarray(dS, dtype=float)
    dt = np.asarray(dt, dtype=float)
    linear = Delta * dS
    convex = 0.5 * G * dS * dS
    if theta is not None:
        time_decay = np.asarray(theta, dtype=float) * dt
    else:
        time_decay = 0.0
    residual = dP - linear - convex - time_decay
    denom = np.abs(linear)
    denom = np.where(denom < 1e-12, np.nan, denom)
    return np.clip(1.0 - np.abs(residual) / denom, 0.0, 1.0)
