from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.data.loader_pc1 import build_residual_matrix

def get_pc1_window_a() -> pd.Series:
    R_df, _ = build_residual_matrix("A")
    R = R_df.values.astype(np.float64)
    Rc = R - R.mean(axis=0)
    _, evc = np.linalg.eigh(Rc.T @ Rc / max(Rc.shape[0] - 1, 1))
    u1 = evc[:, -1]
    pc1 = Rc @ u1
    if np.corrcoef(pc1, np.abs(R).sum(axis=1))[0, 1] < 0:
        pc1 = -pc1
    return pd.Series(pc1, index=R_df.index, name="PC1")

def fetch_yahoo(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"].iloc[:, 0]
    else:
        s = df["Close"]
    s.name = ticker
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s

def realized_vol_21d(spx: pd.Series) -> pd.Series:
    log_ret = np.log(spx / spx.shift(1))
    rv = log_ret.rolling(21).std() * np.sqrt(252) * 100
    rv.name = "RV21"
    return rv

def per_benchmark_stats(pc1: pd.Series, bench: pd.Series) -> dict:
    df = pd.concat([pc1, bench], axis=1).dropna()
    if df.shape[0] < 30:
        return {"n": int(df.shape[0]), "warning": "insufficient overlap"}
    y = df.iloc[:, 0].values.astype(float)
    x = df.iloc[:, 1].values.astype(float)
    dy = np.diff(y); dx = np.diff(x)

    out = {"n": int(df.shape[0]),
           "level_pearson": float(pearsonr(x, y).statistic),
           "level_spearman": float(spearmanr(x, y).statistic),
           "diff_pearson": float(pearsonr(dx, dy).statistic),
           "diff_spearman": float(spearmanr(dx, dy).statistic)}

    try:
        coint_stat, coint_p, _ = coint(y, x, autolag="AIC")
        out["coint_p"] = float(coint_p)
        out["cointegrated"] = bool(coint_p < 0.05)
    except Exception as e:
        out["coint_p"] = None
        out["coint_error"] = str(e)

    X = add_constant(x)
    res = OLS(y, X).fit()
    out["ols_r2"] = float(res.rsquared)
    out["ols_slope"] = float(res.params[1])
    out["ols_intercept"] = float(res.params[0])

    try:
        adf_p = adfuller(res.resid, autolag="AIC")[1]
        out["resid_adf_p"] = float(adf_p)
    except Exception as e:
        out["resid_adf_p"] = None
        out["resid_adf_error"] = str(e)
    return out

def multivariate_stats(pc1: pd.Series, benches: dict) -> dict:
    df = pd.concat([pc1] + list(benches.values()), axis=1).dropna()
    if df.shape[0] < 30:
        return {"n": int(df.shape[0]), "warning": "insufficient overlap"}
    y = df.iloc[:, 0].values.astype(float)
    X_raw = df.iloc[:, 1:].values.astype(float)
    cols = list(df.columns[1:])

    X = add_constant(X_raw)
    res = OLS(y, X).fit()
    out = {"n": int(df.shape[0]),
           "benchmarks": cols,
           "r2": float(res.rsquared),
           "adj_r2": float(res.rsquared_adj),
           "coefs": {c: float(b) for c, b in zip(cols, res.params[1:])},
           "tvalues": {c: float(t) for c, t in zip(cols, res.tvalues[1:])},
           "pvalues": {c: float(p) for c, p in zip(cols, res.pvalues[1:])}}
    try:
        adf_p = adfuller(res.resid, autolag="AIC")[1]
        out["resid_adf_p"] = float(adf_p)
        out["resid_stationary"] = bool(adf_p < 0.05)
    except Exception as e:
        out["resid_adf_p"] = None
        out["resid_adf_error"] = str(e)

    dy = np.diff(y); dX = np.diff(X_raw, axis=0)
    dX_c = add_constant(dX)
    res_d = OLS(dy, dX_c).fit()
    out["diff_r2"] = float(res_d.rsquared)
    out["diff_coefs"] = {c: float(b) for c, b in zip(cols, res_d.params[1:])}
    out["diff_pvalues"] = {c: float(p) for c, p in zip(cols, res_d.pvalues[1:])}
    return out

def main():
    pc1 = get_pc1_window_a()
    pc1.index = pd.to_datetime(pc1.index).tz_localize(None).normalize()
    start = (pc1.index.min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (pc1.index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    print(f"PC1 range: {pc1.index.min().date()} -> {pc1.index.max().date()} (n={len(pc1)})")
    print(f"Yahoo download range: {start} -> {end}")

    tickers = {"VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M",
               "VIX6M": "^VIX6M", "VVIX": "^VVIX", "SKEW": "^SKEW",
               "SPX": "^GSPC"}
    series = {}
    for name, tk in tickers.items():
        s = fetch_yahoo(tk, start, end)
        if not s.empty:
            series[name] = s
            print(f"  {tk}: {len(s)} rows")
        else:
            print(f"  {tk}: EMPTY")

    if "SPX" in series:
        series["RV21"] = realized_vol_21d(series["SPX"])
        del series["SPX"]

    benches = {name: s.reindex(pc1.index).rename(name) for name, s in series.items()}

    results = {"window": "A",
               "pc1_n": int(len(pc1)),
               "date_range": [str(pc1.index.min().date()), str(pc1.index.max().date())],
               "per_benchmark": {}}

    for name, b in benches.items():
        print(f"\n--- {name} ---")
        stats = per_benchmark_stats(pc1, b)
        results["per_benchmark"][name] = stats
        if "warning" in stats:
            print(f"  {stats}")
            continue
        print(f"  n={stats['n']}, level Pearson={stats['level_pearson']:.3f}, "
              f"diff Pearson={stats['diff_pearson']:.3f}, "
              f"OLS R^2={stats['ols_r2']:.3f}, coint p={stats.get('coint_p')}")

    keep = {k: v for k, v in benches.items()
            if results["per_benchmark"][k].get("warning") is None}
    if keep:
        print("\n--- Multivariate OLS PC1 ~ all ---")
        mv = multivariate_stats(pc1, keep)
        results["multivariate"] = mv
        print(f"  n={mv['n']}, R^2={mv['r2']:.3f}, adj R^2={mv['adj_r2']:.3f}, "
              f"resid ADF p={mv.get('resid_adf_p')}")
        print(f"  per-coef p-values: {mv['pvalues']}")
        print(f"  diff R^2={mv['diff_r2']:.3f}")

    out_path = ROOT / "results" / "multi_vol_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")

    panel_df = pd.concat([pc1] + list(benches.values()), axis=1)
    panel_df.index.name = "Date"
    parquet_path = ROOT / "data" / "processed" / "vol_benchmarks.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(parquet_path)
    print(f"saved: {parquet_path}")

if __name__ == "__main__":
    main()
