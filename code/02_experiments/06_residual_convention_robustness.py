from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
RES = ROOT / "results"

from src.data.loader_pc1 import build_residual_matrix
from src.metrics.linearity import L_B
from src.metrics.bsm_greeks import bsm_delta, bsm_call_equivalent_delta
from src.data import config as cfg
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import norm

SEED = 2026
np.random.seed(SEED)

def pc1_phenomenon_stats(R_df: pd.DataFrame) -> dict:
    R = R_df.values.astype(np.float64)
    if R.shape[0] < 30 or R.shape[1] < 3:
        return {"n_dates": int(R.shape[0]), "n_options": int(R.shape[1]),
                "warning": "insufficient observations"}
    Rc = R - R.mean(axis=0)
    cov = Rc.T @ Rc / max(R.shape[0] - 1, 1)
    ev, evc = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    evc = evc[:, order]
    pc1 = Rc @ evc[:, 0]
    if np.corrcoef(pc1, np.abs(R).sum(axis=1))[0, 1] < 0:
        pc1 = -pc1
    var_share = float(ev[0] / ev.sum())
    acf1 = float(np.corrcoef(pc1[:-1], pc1[1:])[0, 1])
    half_life = float(-np.log(2) / np.log(acf1)) if 0 < acf1 < 1 else float("nan")
    try:
        adf_p = float(adfuller(pc1, autolag="AIC")[1])
    except Exception:
        adf_p = float("nan")
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p = float(kpss(pc1, regression="c", nlags="auto")[1])
    except Exception:
        kpss_p = float("nan")
    n_pos = int((evc[:, 0] > 0).sum())
    n_neg = int((evc[:, 0] < 0).sum())
    same_sign_pct = max(n_pos, n_neg) / len(evc[:, 0]) * 100
    return {
        "n_dates": int(R.shape[0]),
        "n_options": int(R.shape[1]),
        "pc1_var_share": var_share,
        "pc1_acf_lag1": acf1,
        "half_life_days": half_life,
        "pc1_adf_p": adf_p,
        "pc1_kpss_p": kpss_p,
        "same_sign_loadings_pct": same_sign_pct,
    }

def run_cross_window():
    out = {}
    for win in ["A", "B", "C"]:
        R_df, _ = build_residual_matrix(win)
        stats = pc1_phenomenon_stats(R_df)
        print(f"  Window {win}: n={stats['n_dates']}d×{stats['n_options']}opts, "
              f"pc1_var={stats.get('pc1_var_share', np.nan):.4f}, "
              f"acf={stats.get('pc1_acf_lag1', np.nan):.4f}, "
              f"adf_p={stats.get('pc1_adf_p', np.nan):.4f}, "
              f"kpss_p={stats.get('pc1_kpss_p', np.nan):.4f}")
        out[f"window_{win}"] = stats
    return out

def build_raw_delta_gap_matrix(window: str = "A") -> pd.DataFrame:
    PROC = cfg.PROCESSED_DIR
    panel = pd.read_parquet(PROC / "options_panel.parquet")
    qr = pd.read_csv(PROC / "quality_report.csv", parse_dates=["expiry"])
    spx_pcp = pd.read_parquet(PROC / "spx_pcp.parquet")
    q_imp = pd.read_parquet(PROC / "q_implied.parquet")
    from src.data.loader_pc1 import (window_a_tickers, window_b_tickers,
                                       window_c_tickers, common_dates)
    if window == "A":
        ta = window_a_tickers(qr)
    elif window == "B":
        ta = window_b_tickers(qr)
    elif window == "C":
        meta = pd.read_parquet(PROC / "options_meta.parquet")
        ta = window_c_tickers(meta, qr)
    cdA = sorted(common_dates(panel, ta))

    sub = panel[panel["ticker"].isin(ta)].dropna(
        subset=["Delta Mid Price", "Implied Volatility Mid"]).copy()
    sub = sub[sub["Date"].dt.date.isin(set(cdA))]
    sub = sub.merge(spx_pcp[["Date", "S_pcp"]], on="Date", how="left")
    sub = sub.merge(q_imp[["Date", "q_implied"]], on="Date", how="left")
    bad_q = (sub["q_implied"] < 0.001) | (sub["q_implied"] > 0.05)
    sub.loc[bad_q, "q_implied"] = np.nan
    sub["q_used"] = sub["q_implied"].fillna(cfg.Q_BASELINE)
    sub["sigma"] = sub["Implied Volatility Mid"] / 100.0
    sub["tau"] = (sub["expiry"] - sub["Date"]).dt.days / 365.25
    sub = sub.dropna(subset=["S_pcp", "tau"]).query("tau > 0").reset_index(drop=True)

    sub["delta_eq_mkt"] = bsm_call_equivalent_delta(
        sub["Delta Mid Price"].values, sub["q_used"].values, sub["tau"].values,
        sub["option_type"].values)
    sub["delta_bsm"] = bsm_delta(
        sub["S_pcp"].values, sub["strike"].values.astype(float),
        cfg.R, sub["q_used"].values, sub["sigma"].values, sub["tau"].values,
        sub["option_type"].values)
    sub["delta_eq_bsm"] = bsm_call_equivalent_delta(
        sub["delta_bsm"].values, sub["q_used"].values, sub["tau"].values,
        sub["option_type"].values)
    bad_eq = (sub["delta_eq_mkt"] < -0.001) | (sub["delta_eq_mkt"] > 1.001)
    sub = sub[~bad_eq].reset_index(drop=True)

    sub["raw_gap"] = sub["delta_eq_mkt"] - sub["delta_eq_bsm"]
    rmat = sub.pivot_table(index="Date", columns="ticker", values="raw_gap",
                            aggfunc="mean").dropna()
    return rmat

def run_raw_vs_LB():
    out = {}
    R_LB, _ = build_residual_matrix("A")
    R_raw = build_raw_delta_gap_matrix("A")

    common_dates_idx = R_LB.index.intersection(R_raw.index)
    common_tickers = sorted(set(R_LB.columns) & set(R_raw.columns))
    R_LB = R_LB.loc[common_dates_idx, common_tickers]
    R_raw = R_raw.loc[common_dates_idx, common_tickers]
    print(f"  Aligned: {R_LB.shape[0]} dates × {R_LB.shape[1]} options")

    s_lb = pc1_phenomenon_stats(R_LB)
    s_raw = pc1_phenomenon_stats(R_raw)
    out["L_B_residual"] = s_lb
    out["raw_delta_gap"] = s_raw

    def _pc1(R):
        Rc = R - R.mean(axis=0)
        cov = Rc.T @ Rc / max(Rc.shape[0]-1, 1)
        ev, evc = np.linalg.eigh(cov)
        order = np.argsort(ev)[::-1]
        u1 = evc[:, order[0]]
        pc1 = Rc @ u1
        if np.corrcoef(pc1, np.abs(R).sum(axis=1))[0, 1] < 0:
            pc1 = -pc1
        return pc1

    pc1_lb = _pc1(R_LB.values)
    pc1_raw = _pc1(R_raw.values)
    pc1_pearson = float(np.corrcoef(pc1_lb, pc1_raw)[0, 1])
    out["pc1_pearson_LB_vs_raw"] = pc1_pearson
    out["abs_pc1_pearson"] = abs(pc1_pearson)
    print(f"  PC1 Pearson (L_B vs raw): {pc1_pearson:.4f}")
    print(f"  L_B  PC1: var_share={s_lb['pc1_var_share']:.4f}, "
          f"acf={s_lb['pc1_acf_lag1']:.4f}, "
          f"adf_p={s_lb['pc1_adf_p']:.4f}")
    print(f"  raw  PC1: var_share={s_raw['pc1_var_share']:.4f}, "
          f"acf={s_raw['pc1_acf_lag1']:.4f}, "
          f"adf_p={s_raw['pc1_adf_p']:.4f}")
    return out

def run_dm_test():
    real = json.loads((RES / "real_domain_results.json").read_text())
    abl = json.loads((RES / "ablation_results.json").read_text())
    rs = real["summary"]; abl_s = abl["summary"]

    g11 = real["all_models"]["garch"]["(1,1)"]
    g11_pf = {f: float(g11[f]["test_auc"]) for f in ["3","4","5"]}
    rs["garch"] = {"per_fold_test": g11_pf,
                   "median_test": float(np.median(list(g11_pf.values())))}

    nolt_pf = abl_s["nolt_no_sequence"]["per_fold_test"]
    nolt_aucs = np.array([nolt_pf[f] for f in ["3","4","5"]])

    baselines = [
        ("BSM", rs["bsm_threshold"]["per_fold_test"]),
        ("GARCH(1,1)", rs["garch"]["per_fold_test"]),
        ("XGBoost", rs["xgboost"]["per_fold_test"]),
        ("LSTM", rs["lstm_single"]["per_fold_test"]),
        ("NOLT with lookback", rs["nolt"]["per_fold_test"]),
    ]

    out = {"folds": ["3","4","5"], "nolt_per_fold": nolt_aucs.tolist(), "comparisons": {}}
    p_values_one_sided = []
    for name, pf in baselines:
        bauc = np.array([pf[f] for f in ["3","4","5"]])
        diff = nolt_aucs - bauc

        dm_mean = float(diff.mean())
        dm_sd = float(diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 1e-12
        dm_t = dm_mean / (dm_sd / np.sqrt(len(diff)))

        from scipy.stats import t as student_t
        p = float(1.0 - student_t.cdf(dm_t, df=len(diff)-1))
        p_values_one_sided.append(p)
        out["comparisons"][name] = {
            "baseline_per_fold": bauc.tolist(),
            "diff_per_fold": diff.tolist(),
            "diff_mean": dm_mean,
            "diff_std": dm_sd,
            "dm_t": dm_t,
            "p_value_one_sided": p,
        }
        print(f"  NOLT vs {name:<22}: diff_mean={dm_mean:+.4f}, t={dm_t:.3f}, p={p:.4f}")

    n = len(p_values_one_sided)
    sorted_idx = np.argsort(p_values_one_sided)
    sorted_p = np.array(p_values_one_sided)[sorted_idx]
    holm_adj = np.zeros(n)
    for k, p in enumerate(sorted_p):
        holm_adj[k] = min((n - k) * p, 1.0)

    for k in range(1, n):
        holm_adj[k] = max(holm_adj[k], holm_adj[k-1])

    holm_back = np.zeros(n)
    holm_back[sorted_idx] = holm_adj
    for (name, _), p_holm in zip(baselines, holm_back):
        out["comparisons"][name]["p_value_holm_adjusted"] = float(p_holm)
        print(f"  NOLT vs {name:<22}: Holm-adjusted p = {p_holm:.4f}")
    return out

def main():
    out = {}
    print("=" * 70)
    print("Phase 1 robustness experiments (seed=2026)")
    print("=" * 70)

    print("\n[1/3] Cross-window phenomenon stats (Windows A, B, C)")
    out["cross_window"] = run_cross_window()

    print("\n[2/3] Raw delta-gap PCA vs L_B PCA (Window A)")
    out["raw_vs_LB"] = run_raw_vs_LB()

    print("\n[3/3] Diebold-Mariano test + Holm correction")
    out["dm_test"] = run_dm_test()

    mvb = json.loads((RES / "multi_vol_benchmark.json").read_text())
    out["vix_family_individual_R2"] = {
        name: stats.get("ols_r2") for name, stats in mvb["per_benchmark"].items()
    }
    out["vix_family_multivariate"] = {
        "level_R2": mvb["multivariate"]["r2"],
        "diff_R2": mvb["multivariate"]["diff_r2"],
        "resid_adf_p": mvb["multivariate"]["resid_adf_p"],
    }

    out_path = RES / "phase1_robustness.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_path}")

if __name__ == "__main__":
    main()
