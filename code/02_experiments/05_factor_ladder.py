from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.data.loader_pc1 import build_residual_matrix

def get_pc1() -> pd.Series:
    R_df, _ = build_residual_matrix("A")
    R = R_df.values.astype(np.float64)
    Rc = R - R.mean(axis=0)
    _, evc = np.linalg.eigh(Rc.T @ Rc / max(Rc.shape[0] - 1, 1))
    pc1 = Rc @ evc[:, -1]
    if np.corrcoef(pc1, np.abs(R).sum(axis=1))[0, 1] < 0:
        pc1 = -pc1
    s = pd.Series(pc1, index=R_df.index, name="PC1")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s

def fetch(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    s = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s.name = ticker
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s

def realized_moment(spx: pd.Series, window: int, kind: str) -> pd.Series:
    log_ret = np.log(spx / spx.shift(1))
    if kind == "vol":
        return (log_ret.rolling(window).std() * np.sqrt(252) * 100).rename(f"RV{window}")
    if kind == "skew":
        return log_ret.rolling(window).skew().rename(f"RSkew{window}")
    if kind == "kurt":
        return log_ret.rolling(window).kurt().rename(f"RKurt{window}")
    raise ValueError(kind)

def try_fama_french(start: str, end: str) -> pd.DataFrame | None:
    try:
        from pandas_datareader.data import DataReader
        ff3 = DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                         start=start, end=end)[0]
        mom = DataReader("F-F_Momentum_Factor_daily", "famafrench",
                         start=start, end=end)[0]
        df = ff3.join(mom, how="inner")
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.columns = [c.strip().replace(" ", "") for c in df.columns]
        df = df[["Mkt-RF", "SMB", "HML", "Mom"]].rename(
            columns={"Mkt-RF": "MktRF", "Mom": "MOM"})
        return df
    except Exception as e:
        print(f"  Fama-French fetch failed: {e}")
        return None

def fit_block(y: np.ndarray, X_raw: np.ndarray, dy: np.ndarray, dX_raw: np.ndarray,
              cols: list[str]) -> dict:
    X = add_constant(X_raw); res = OLS(y, X).fit()
    dX = add_constant(dX_raw); res_d = OLS(dy, dX).fit()

    out = {"n_vars": len(cols), "vars": cols,
           "r2_level": float(res.rsquared),
           "adj_r2_level": float(res.rsquared_adj),
           "r2_diff": float(res_d.rsquared),
           "adj_r2_diff": float(res_d.rsquared_adj)}
    try:
        out["resid_adf_p_level"] = float(adfuller(res.resid, autolag="AIC")[1])
    except Exception:
        out["resid_adf_p_level"] = None
    return out

def main():
    pc1 = get_pc1()
    start = (pc1.index.min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = (pc1.index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    print(f"PC1: {pc1.index.min().date()} -> {pc1.index.max().date()}, n={len(pc1)}")

    print("\n[1/3] Yahoo download...")
    yahoo_tickers = {
        "VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M",
        "VVIX": "^VVIX", "SKEW": "^SKEW",
        "TNX": "^TNX", "IRX": "^IRX",
        "HYG": "HYG", "LQD": "LQD",
        "UUP": "UUP",
        "SPX": "^GSPC",
    }
    series = {}
    for n, tk in yahoo_tickers.items():
        s = fetch(tk, start, end)
        if not s.empty:
            s.name = n
            series[n] = s
            print(f"  {tk}: {len(s)} rows")
        else:
            print(f"  {tk}: EMPTY")

    if "SPX" in series:
        spx = series.pop("SPX")
        series["RV21"] = realized_moment(spx, 21, "vol")
        series["RSkew21"] = realized_moment(spx, 21, "skew")
        series["RKurt21"] = realized_moment(spx, 21, "kurt")
        print(f"  RV21, RSkew21, RKurt21 computed from SPX")

    print("\n[2/3] Fama-French download...")
    ff = try_fama_french(start, end)
    if ff is not None:
        for c in ff.columns:
            series[c] = ff[c].rename(c)
        print(f"  Fama-French + MOM: {ff.shape[0]} rows, cols={list(ff.columns)}")

    print("\n[3/3] Build aligned panel...")
    panel = pd.concat([pc1] + [s.reindex(pc1.index).rename(s.name) for s in series.values()],
                      axis=1).dropna()
    print(f"  Aligned panel: {panel.shape}")

    BLOCKS = {
        "S0_VIX_only": ["VIX"],
        "S1_vol_surface": ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "RV21"],
        "S2_plus_rates": ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "RV21",
                          "TNX", "IRX"],
        "S3_plus_credit": ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "RV21",
                           "TNX", "IRX", "HYG", "LQD"],
        "S4_plus_fx": ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "RV21",
                       "TNX", "IRX", "HYG", "LQD", "UUP"],
        "S5_plus_higher_moments": ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "RV21",
                                   "TNX", "IRX", "HYG", "LQD", "UUP",
                                   "RSkew21", "RKurt21"],
    }
    if ff is not None:
        BLOCKS["S6_plus_fama_french"] = (BLOCKS["S5_plus_higher_moments"]
                                          + ["MktRF", "SMB", "HML", "MOM"])

    y = panel["PC1"].values.astype(float)
    dy = np.diff(y)

    results = {"window": "A",
               "pc1_n": int(len(panel)),
               "date_range": [str(panel.index.min().date()), str(panel.index.max().date())],
               "specifications": {}}

    print("\n--- Multivariate specifications ---")
    print(f"{'Spec':<28} {'k':>3} {'R² lvl':>8} {'R² Δ':>8} {'resid ADF p':>14}")
    print("-" * 72)
    for name, cols in BLOCKS.items():
        usable = [c for c in cols if c in panel.columns]
        if len(usable) < len(cols):
            print(f"  {name}: missing {set(cols) - set(usable)}; using {usable}")
        X_raw = panel[usable].values.astype(float)
        dX_raw = np.diff(X_raw, axis=0)
        out = fit_block(y, X_raw, dy, dX_raw, usable)
        results["specifications"][name] = out
        adf_str = f"{out['resid_adf_p_level']:.4f}" if out['resid_adf_p_level'] is not None else "n/a"
        print(f"{name:<28} {out['n_vars']:>3} {out['r2_level']:>8.3f} "
              f"{out['r2_diff']:>8.3f} {adf_str:>14}")

    out_path = ROOT / "results" / "full_factor_robustness.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")

    full_panel = panel.copy()
    full_panel.index.name = "Date"
    parquet_path = ROOT / "data" / "processed" / "full_factor_panel.parquet"
    full_panel.to_parquet(parquet_path)
    print(f"saved: {parquet_path}")

if __name__ == "__main__":
    main()
