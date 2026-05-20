from __future__ import annotations
import json, sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss

SEED = 2026
np.random.seed(SEED)
RES = ROOT / "results"

def main():

    vb = pd.read_parquet(ROOT / "data" / "processed" / "vol_benchmarks.parquet")
    vb.index = pd.to_datetime(vb.index).normalize()
    panel = pd.read_parquet(ROOT / "data" / "processed" / "options_panel.parquet")
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.normalize()
    spx = pd.read_parquet(ROOT / "data" / "processed" / "spx_pcp.parquet")
    spx["Date"] = pd.to_datetime(spx["Date"]).dt.normalize()

    vb["VRP"] = vb["VIX"]**2 - vb["RV21"]**2

    vb["TS"] = vb["VIX3M"] - vb["VIX9D"]

    panel = panel.merge(spx[["Date", "S_pcp"]], on="Date", how="left")
    panel["m"] = np.log(panel["strike"].astype(float) / panel["S_pcp"])
    panel = panel.dropna(subset=["Implied Volatility Mid", "S_pcp"])

    panel = panel[(panel["Date"] >= vb.index.min()) & (panel["Date"] <= vb.index.max())]

    otm_puts = panel[(panel["option_type"] == "P") & (panel["m"] < 0)]
    otm_calls = panel[(panel["option_type"] == "C") & (panel["m"] > 0)]
    daily_otm_put_iv = otm_puts.groupby("Date")["Implied Volatility Mid"].mean()
    daily_otm_call_iv = otm_calls.groupby("Date")["Implied Volatility Mid"].mean()
    skew_a = (daily_otm_put_iv - daily_otm_call_iv).rename("SkewA")
    skew_a.index = pd.to_datetime(skew_a.index).normalize()

    df = vb[["PC1", "VRP", "SKEW", "TS"]].join(skew_a, how="inner").dropna()
    print(f"Aligned panel: {len(df)} dates  cols={list(df.columns)}")
    print(f"  VRP   mean={df['VRP'].mean():.2f}, std={df['VRP'].std():.2f}")
    print(f"  SKEW  mean={df['SKEW'].mean():.2f}, std={df['SKEW'].std():.2f}")
    print(f"  TS    mean={df['TS'].mean():.2f}, std={df['TS'].std():.2f}")
    print(f"  SkewA mean={df['SkewA'].mean():.2f}, std={df['SkewA'].std():.2f}")

    out = {"seed": SEED, "n_obs": int(len(df)),
           "variables": ["VRP", "SKEW", "TS", "SkewA"], "level": {}, "diff": {}}

    def fit_ols(y, X):
        Xc = add_constant(X)
        res = OLS(y.astype(float), Xc.astype(float)).fit()
        return res

    print("\n== Level regression ==")
    y = df["PC1"].values
    indiv_lvl = {}
    for v in ["VRP", "SKEW", "TS", "SkewA"]:
        x = df[v].values.reshape(-1, 1)
        r = fit_ols(y, x)
        indiv_lvl[v] = {"r2": float(r.rsquared), "coef": float(r.params[1]),
                         "tstat": float(r.tvalues[1]), "p": float(r.pvalues[1]),
                         "std_coef": float(r.params[1] * df[v].std() / df["PC1"].std())}
        print(f"  {v:<6}: R^2={r.rsquared:.4f}, coef={r.params[1]:+.4e}, "
              f"t={r.tvalues[1]:+.3f}, p={r.pvalues[1]:.4f}")

    Xj = df[["VRP", "SKEW", "TS", "SkewA"]].values
    rj = fit_ols(y, Xj)
    joint_lvl = {"r2": float(rj.rsquared), "adj_r2": float(rj.rsquared_adj),
                  "coefs": {v: float(rj.params[i+1]) for i, v in enumerate(["VRP","SKEW","TS","SkewA"])},
                  "tvalues": {v: float(rj.tvalues[i+1]) for i, v in enumerate(["VRP","SKEW","TS","SkewA"])},
                  "pvalues": {v: float(rj.pvalues[i+1]) for i, v in enumerate(["VRP","SKEW","TS","SkewA"])}}
    print(f"  Joint:  R^2={rj.rsquared:.4f}, adj R^2={rj.rsquared_adj:.4f}")

    resid_lvl = np.asarray(rj.resid)
    try:
        adf_p_lvl = float(adfuller(resid_lvl, autolag="AIC")[1])
    except Exception:
        adf_p_lvl = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p_lvl = float(kpss(resid_lvl, regression="c", nlags="auto")[1])
    except Exception:
        kpss_p_lvl = float("nan")
    out["level"] = {"individual": indiv_lvl, "joint": joint_lvl,
                     "resid_adf_p": adf_p_lvl, "resid_kpss_p": kpss_p_lvl}
    print(f"  resid ADF p={adf_p_lvl:.4f}, KPSS p={kpss_p_lvl:.4f}")

    print("\n== First-difference regression (the gap mechanism) ==")
    df_d = df.diff().dropna()
    y_d = df_d["PC1"].values
    indiv_dif = {}
    for v in ["VRP", "SKEW", "TS", "SkewA"]:
        x = df_d[v].values.reshape(-1, 1)
        r = fit_ols(y_d, x)
        indiv_dif[v] = {"r2": float(r.rsquared), "coef": float(r.params[1]),
                         "tstat": float(r.tvalues[1]), "p": float(r.pvalues[1]),
                         "std_coef": float(r.params[1] * df_d[v].std() / df_d["PC1"].std())}
        print(f"  {v:<6}: R^2={r.rsquared:.4f}, coef={r.params[1]:+.4e}, "
              f"t={r.tvalues[1]:+.3f}, p={r.pvalues[1]:.4f}")
    Xjd = df_d[["VRP", "SKEW", "TS", "SkewA"]].values
    rjd = fit_ols(y_d, Xjd)
    joint_dif = {"r2": float(rjd.rsquared), "adj_r2": float(rjd.rsquared_adj),
                  "coefs": {v: float(rjd.params[i+1]) for i, v in enumerate(["VRP","SKEW","TS","SkewA"])},
                  "tvalues": {v: float(rjd.tvalues[i+1]) for i, v in enumerate(["VRP","SKEW","TS","SkewA"])},
                  "pvalues": {v: float(rjd.pvalues[i+1]) for i, v in enumerate(["VRP","SKEW","TS","SkewA"])}}
    print(f"  Joint:  R^2={rjd.rsquared:.4f}, adj R^2={rjd.rsquared_adj:.4f}")
    resid_dif = np.asarray(rjd.resid)
    try:
        adf_p_dif = float(adfuller(resid_dif, autolag="AIC")[1])
    except Exception:
        adf_p_dif = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p_dif = float(kpss(resid_dif, regression="c", nlags="auto")[1])
    except Exception:
        kpss_p_dif = float("nan")
    out["diff"] = {"individual": indiv_dif, "joint": joint_dif,
                    "resid_adf_p": adf_p_dif, "resid_kpss_p": kpss_p_dif}
    print(f"  resid ADF p={adf_p_dif:.4f}, KPSS p={kpss_p_dif:.4f}")

    fitted_lvl_full = rj.fittedvalues
    fitted_dif_full = rjd.fittedvalues
    out["fitted_level_dates"] = [d.isoformat() for d in df.index]
    out["fitted_level_values"] = [float(v) for v in fitted_lvl_full]
    out["pc1_level_values"] = [float(v) for v in df["PC1"].values]
    out["fitted_diff_dates"] = [d.isoformat() for d in df_d.index]
    out["fitted_diff_values"] = [float(v) for v in fitted_dif_full]
    out["pc1_diff_values"] = [float(v) for v in y_d]

    out_path = RES / "phase5_risk_premium.json"
    out_path.write_text(json.dumps(out, indent=2,
                                      default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_path}")

if __name__ == "__main__":
    main()
