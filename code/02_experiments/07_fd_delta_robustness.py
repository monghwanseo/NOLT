from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from src.data import config as cfg
from src.data.tasks import window_a_tickers, common_dates
from src.metrics.bsm_greeks import bsm_delta, bsm_call_equivalent_delta
from src.metrics.linearity import L_B

def _safe_inv_norm(p, clip_eps=1e-6):
    pc = np.clip(p, clip_eps, 1.0 - clip_eps)
    return norm.ppf(pc)

def compute_fd_delta(panel_sub: pd.DataFrame):
    panel_sub = panel_sub.sort_values(["Date", "expiry", "option_type", "strike"]).reset_index(drop=True)
    panel_sub["d_fd_dual"] = np.nan
    panel_sub["d_fd_shifted"] = np.nan
    panel_sub["fd_kind"] = "none"

    grouped = panel_sub.groupby(["Date", "expiry", "option_type"], sort=False)
    for (date, expiry, otype), g in grouped:
        if len(g) < 2:
            continue
        K = g["strike"].values.astype(float)
        C = g["Mid Price"].values.astype(float)
        sigma_iv = g["sigma"].values.astype(float)
        tau = g["tau"].values.astype(float)

        n = len(K)
        dCdK = np.full(n, np.nan)
        kind = np.array(["none"] * n, dtype=object)

        for i in range(1, n - 1):
            if np.isfinite(C[i - 1]) and np.isfinite(C[i + 1]):
                dCdK[i] = (C[i + 1] - C[i - 1]) / (K[i + 1] - K[i - 1])
                kind[i] = "centered"

        if n >= 2 and np.isfinite(C[0]) and np.isfinite(C[1]):
            dCdK[0] = (C[1] - C[0]) / (K[1] - K[0])
            kind[0] = "forward"

        if n >= 2 and np.isfinite(C[-1]) and np.isfinite(C[-2]):
            dCdK[-1] = (C[-1] - C[-2]) / (K[-1] - K[-2])
            kind[-1] = "backward"

        r = cfg.R
        disc = np.exp(r * tau)
        if otype == "C":
            d_dual = -disc * dCdK
        else:
            d_dual = 1.0 - disc * dCdK
        d_dual = np.clip(d_dual, 0.0, 1.0)

        shift = sigma_iv * np.sqrt(np.maximum(tau, 1e-12))
        d_shifted = norm.cdf(_safe_inv_norm(d_dual) + shift)
        d_shifted = np.clip(d_shifted, 0.0, 1.0)

        panel_sub.loc[g.index, "d_fd_dual"] = d_dual
        panel_sub.loc[g.index, "d_fd_shifted"] = d_shifted
        panel_sub.loc[g.index, "fd_kind"] = kind

    return panel_sub

def build_residual_matrix_variant(delta_eq_col: str):
    PROC = cfg.PROCESSED_DIR
    panel = pd.read_parquet(PROC / "options_panel.parquet")
    qr = pd.read_csv(PROC / "quality_report.csv", parse_dates=["expiry"])
    spx_pcp = pd.read_parquet(PROC / "spx_pcp.parquet")
    q_imp = pd.read_parquet(PROC / "q_implied.parquet")

    ta = window_a_tickers(qr)
    cdA = sorted(common_dates(panel, ta))

    sub = panel[panel["ticker"].isin(ta)].dropna(
        subset=["Delta Mid Price", "Implied Volatility Mid", "Mid Price"]).copy()
    sub = sub[sub["Date"].dt.date.isin(set(cdA))]
    sub = sub.merge(spx_pcp[["Date", "S_pcp"]], on="Date", how="left")
    sub = sub.merge(q_imp[["Date", "q_implied"]], on="Date", how="left")

    bad_q = (sub["q_implied"] < 0.001) | (sub["q_implied"] > 0.05)
    sub.loc[bad_q, "q_implied"] = np.nan
    sub["q_used"] = sub["q_implied"].fillna(cfg.Q_BASELINE)
    sub["sigma"] = sub["Implied Volatility Mid"] / 100.0
    sub["tau"] = (sub["expiry"] - sub["Date"]).dt.days / 365.25
    sub = sub.dropna(subset=["S_pcp", "tau"]).query("tau > 0").reset_index(drop=True)

    sub = compute_fd_delta(sub)

    if delta_eq_col == "vendor":
        sub["delta_eq_mkt"] = bsm_call_equivalent_delta(
            sub["Delta Mid Price"].values, sub["q_used"].values,
            sub["tau"].values, sub["option_type"].values)
    elif delta_eq_col == "dual":

        sub["delta_eq_mkt"] = sub["d_fd_dual"].values
    elif delta_eq_col == "shifted":
        sub["delta_eq_mkt"] = sub["d_fd_shifted"].values
    else:
        raise ValueError(delta_eq_col)

    sub["delta_bsm"] = bsm_delta(
        sub["S_pcp"].values, sub["strike"].values.astype(float),
        cfg.R, sub["q_used"].values, sub["sigma"].values,
        sub["tau"].values, sub["option_type"].values)
    sub["delta_eq_bsm"] = bsm_call_equivalent_delta(
        sub["delta_bsm"].values, sub["q_used"].values,
        sub["tau"].values, sub["option_type"].values)

    bad_eq = (sub["delta_eq_mkt"] < -0.001) | (sub["delta_eq_mkt"] > 1.001)
    sub = sub[~bad_eq].reset_index(drop=True)
    sub = sub.dropna(subset=["delta_eq_mkt", "delta_eq_bsm"]).reset_index(drop=True)

    sub["LB_mkt"] = L_B(sub["delta_eq_mkt"].values)
    sub["LB_bsm"] = L_B(sub["delta_eq_bsm"].values)
    sub["residual"] = sub["LB_mkt"] - sub["LB_bsm"]

    rmat = sub.pivot_table(index="Date", columns="ticker", values="residual",
                            aggfunc="mean").dropna()
    return rmat, sub

def pca_summary(R_df: pd.DataFrame):
    R = R_df.values.astype(np.float64)
    Rc = R - R.mean(axis=0, keepdims=True)
    cov = Rc.T @ Rc / max(R.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    u1 = eigvecs[:, 0]
    pc1 = Rc @ u1

    if np.corrcoef(pc1, np.abs(R).sum(axis=1))[0, 1] < 0:
        pc1 = -pc1
        u1 = -u1
    var_share = float(eigvals[0] / max(eigvals.sum(), 1e-30))
    acf1 = float(np.corrcoef(pc1[:-1], pc1[1:])[0, 1])
    return {"pc1": pc1, "loadings": u1, "var_share": var_share, "acf1": acf1,
            "dates": R_df.index, "tickers": list(R_df.columns)}

def align_and_compare(stats_base, stats_var):
    common_dates = stats_base["dates"].intersection(stats_var["dates"])
    common_tickers = sorted(set(stats_base["tickers"]) & set(stats_var["tickers"]))

    base_idx_dates = stats_base["dates"].isin(common_dates)
    base_idx_t = [i for i, t in enumerate(stats_base["tickers"]) if t in common_tickers]
    var_idx_dates = stats_var["dates"].isin(common_dates)
    var_idx_t = [i for i, t in enumerate(stats_var["tickers"]) if t in common_tickers]
    return {
        "n_common_dates": int(len(common_dates)),
        "n_common_tickers": int(len(common_tickers)),
    }

def compare_pc1_loadings(R_base: pd.DataFrame, R_var: pd.DataFrame):
    common_dates = R_base.index.intersection(R_var.index)
    common_tickers = sorted(set(R_base.columns) & set(R_var.columns))
    Rb = R_base.loc[common_dates, common_tickers]
    Rv = R_var.loc[common_dates, common_tickers]

    sb = pca_summary(Rb)
    sv = pca_summary(Rv)

    pc1_pearson = float(np.corrcoef(sb["pc1"], sv["pc1"])[0, 1])
    loading_cos = float(np.dot(sb["loadings"], sv["loadings"])
                         / (np.linalg.norm(sb["loadings"]) * np.linalg.norm(sv["loadings"]) + 1e-30))
    return {
        "n_dates": int(Rb.shape[0]),
        "n_tickers": int(Rb.shape[1]),
        "var_share_base": sb["var_share"],
        "var_share_var": sv["var_share"],
        "acf1_base": sb["acf1"],
        "acf1_var": sv["acf1"],
        "pc1_pearson": pc1_pearson,
        "loading_cosine": loading_cos,
    }

def main():
    print("=" * 78)
    print("E1 — Price-implied FD Delta residual robustness")
    print("Variants: DUAL (vendor-free), SHIFTED (sigma*sqrt(tau) corrected)")
    print("=" * 78)

    print("\n[Baseline] vendor Delta")
    R_base, sub_base = build_residual_matrix_variant("vendor")
    print(f"  shape: {R_base.shape}")

    print("\n[Variant A] DUAL — vendor-free dual delta")
    R_dual, sub_dual = build_residual_matrix_variant("dual")
    print(f"  shape: {R_dual.shape}")
    print("  FD kind counts:", sub_dual["fd_kind"].value_counts().to_dict())

    print("\n[Variant B] SHIFTED — dual + sigma*sqrt(tau) shift to real Delta proxy")
    R_shift, sub_shift = build_residual_matrix_variant("shifted")
    print(f"  shape: {R_shift.shape}")

    comp_dual = compare_pc1_loadings(R_base, R_dual)
    comp_shift = compare_pc1_loadings(R_base, R_shift)

    print("\n=== Comparison (FD vs baseline) ===")
    for tag, comp in [("DUAL", comp_dual), ("SHIFTED", comp_shift)]:
        print(f"  {tag}: n_dates={comp['n_dates']}, n_tickers={comp['n_tickers']}")
        print(f"    var_share: base={comp['var_share_base']:.4f}, "
              f"var={comp['var_share_var']:.4f}")
        print(f"    ACF(1)  : base={comp['acf1_base']:.4f}, "
              f"var={comp['acf1_var']:.4f}")
        print(f"    Pearson(PC1_FD, PC1_baseline) = {comp['pc1_pearson']:+.4f}")
        print(f"    Loading cosine similarity     = {comp['loading_cosine']:+.4f}")

    rows = []

    base_stats = pca_summary(R_base)
    rows.append({
        "Variant": "vendor (baseline)",
        "PC1 var share": base_stats["var_share"],
        "PC1 ACF(1)": base_stats["acf1"],
        "Pearson(PC1_FD, PC1_baseline)": 1.0,
        "Loading cosine vs baseline": 1.0,
    })
    rows.append({
        "Variant": "FD dual delta (vendor-free)",
        "PC1 var share": comp_dual["var_share_var"],
        "PC1 ACF(1)": comp_dual["acf1_var"],
        "Pearson(PC1_FD, PC1_baseline)": comp_dual["pc1_pearson"],
        "Loading cosine vs baseline": comp_dual["loading_cosine"],
    })
    rows.append({
        "Variant": "FD shifted delta (real-delta proxy)",
        "PC1 var share": comp_shift["var_share_var"],
        "PC1 ACF(1)": comp_shift["acf1_var"],
        "Pearson(PC1_FD, PC1_baseline)": comp_shift["pc1_pearson"],
        "Loading cosine vs baseline": comp_shift["loading_cosine"],
    })
    df = pd.DataFrame(rows)
    out_csv = ROOT / "paper" / "tables" / "T_residual_robustness.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nsaved: {out_csv}")

    out_json = ROOT / "results" / "E1_fd_delta.json"
    out_json.write_text(json.dumps({
        "baseline_shape": list(R_base.shape),
        "dual_shape": list(R_dual.shape),
        "shifted_shape": list(R_shift.shape),
        "fd_kind_counts_dual": sub_dual["fd_kind"].value_counts().to_dict(),
        "comparison_dual": comp_dual,
        "comparison_shifted": comp_shift,
    }, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"saved: {out_json}")

if __name__ == "__main__":
    main()
