from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

def acf_at_lags(x: np.ndarray, lags: list[int]) -> dict:
    x = x - x.mean()
    denom = (x * x).sum()
    return {f"lag_{k}": float((x[:-k] * x[k:]).sum() / max(denom, 1e-30)) for k in lags}

def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "full_factor_panel.parquet")
    panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
    rob = json.loads((ROOT / "results" / "full_factor_robustness.json").read_text())
    s6_cols = rob["specifications"]["S6_plus_fama_french"]["vars"]
    df = panel.dropna()

    y = df["PC1"].values.astype(float)
    X_raw = df[s6_cols].values.astype(float)
    X = add_constant(X_raw)
    res_lvl = OLS(y, X).fit()
    resid_lvl = res_lvl.resid

    dy = np.diff(y)
    dX_raw = np.diff(X_raw, axis=0)
    dX = add_constant(dX_raw)
    res_dif = OLS(dy, dX).fit()
    resid_dif = res_dif.resid

    lags_short = [1, 2, 3, 5, 10, 21]
    out = {
        "n_panel": int(len(df)),
        "n_factors": len(s6_cols),
        "level_residual": {
            "r2_explained": float(res_lvl.rsquared),
            "r2_unexplained": float(1 - res_lvl.rsquared),
            "var_resid": float(resid_lvl.var(ddof=1)),
            "acf": acf_at_lags(resid_lvl, lags_short),
        },
        "diff_residual": {
            "r2_explained": float(res_dif.rsquared),
            "r2_unexplained": float(1 - res_dif.rsquared),
            "var_resid": float(resid_dif.var(ddof=1)),
            "acf": acf_at_lags(resid_dif, lags_short),
        },
    }

    for label, x in [("level_residual", resid_lvl), ("diff_residual", resid_dif)]:
        try:
            lb = acorr_ljungbox(x, lags=[1, 5, 10, 21], return_df=True)
            out[label]["ljungbox"] = {f"Q_{int(L)}": {"stat": float(s), "p": float(p)}
                                       for L, s, p in zip(lb.index, lb["lb_stat"], lb["lb_pvalue"])}
        except Exception as e:
            out[label]["ljungbox_error"] = str(e)

    print("=== Level residual (after 18-factor fit) ===")
    print(f"  Var unexplained: {out['level_residual']['r2_unexplained']:.3f}")
    print(f"  ACF(1) = {out['level_residual']['acf']['lag_1']:.3f}")
    print(f"  ACF(5) = {out['level_residual']['acf']['lag_5']:.3f}")
    print(f"  ACF(10) = {out['level_residual']['acf']['lag_10']:.3f}")
    print(f"  ACF(21) = {out['level_residual']['acf']['lag_21']:.3f}")
    if "ljungbox" in out["level_residual"]:
        for L, d in out["level_residual"]["ljungbox"].items():
            print(f"  LjungBox {L}: stat={d['stat']:.2f}, p={d['p']:.6f}")

    print("\n=== Diff residual (after 18-factor diff fit) ===")
    print(f"  Var unexplained: {out['diff_residual']['r2_unexplained']:.3f}")
    print(f"  ACF(1) = {out['diff_residual']['acf']['lag_1']:.3f}")
    print(f"  ACF(5) = {out['diff_residual']['acf']['lag_5']:.3f}")
    print(f"  ACF(10) = {out['diff_residual']['acf']['lag_10']:.3f}")
    print(f"  ACF(21) = {out['diff_residual']['acf']['lag_21']:.3f}")
    if "ljungbox" in out["diff_residual"]:
        for L, d in out["diff_residual"]["ljungbox"].items():
            print(f"  LjungBox {L}: stat={d['stat']:.2f}, p={d['p']:.6f}")

    out_path = ROOT / "results" / "residual_structure.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")

if __name__ == "__main__":
    main()
