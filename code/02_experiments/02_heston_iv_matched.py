from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.synthetic.heston import (
    HestonParams, simulate_heston_qe, bsm_delta, call_equiv, L_B,
    make_window_a_universe, acf1, adf_p, kpss_p,
)
from src.synthetic.iv_match import (
    implied_vol_newton, heston_price_per_day, heston_delta_per_day_FD,
)

SEED = 2026
OUT_JSON = ROOT / "results" / "heston_iv_matched.json"

def run_heston_iv_matched(params, n_paths=40, T_total_days=348,
                          S0=100.0, r=0.04, q=0.0117, seed=SEED):
    rng = np.random.default_rng(seed)
    K_arr, T_exp_arr, type_arr = make_window_a_universe(S0)
    N = len(K_arr)

    print(f"  Sim Heston QE: {n_paths}x{T_total_days}", flush=True)
    S, V = simulate_heston_qe(params, S0, r, q, T_total_days, n_paths, rng)
    T_full = T_total_days + 1

    R = np.zeros((n_paths, T_full, N), dtype=np.float64)
    sigma_iv_panel = np.full_like(R, np.nan)

    for day in range(T_full):
        if day % 50 == 0:
            print(f"    day {day}/{T_full - 1}", flush=True)
        S_d = S[:, day]
        V_d = V[:, day]
        days_elapsed_year = day / 365.25
        for T_uniq in np.unique(T_exp_arr):
            mask = T_exp_arr == T_uniq
            tau = float(T_uniq - days_elapsed_year)
            tau = max(tau, 1.0 / 365.0)
            K_sub = K_arr[mask]
            type_sub = type_arr[mask]

            price = heston_price_per_day(S_d, V_d, K_sub, tau, type_sub, r, q, params)
            delta_market = heston_delta_per_day_FD(S_d, V_d, K_sub, tau, type_sub, r, q, params)

            S_b = np.broadcast_to(S_d[:, None], price.shape)
            K_b = np.broadcast_to(K_sub[None, :], price.shape)
            tau_b = np.broadcast_to(tau, price.shape)
            otype_b = np.broadcast_to(type_sub[None, :], price.shape)
            sigma_init = float(np.sqrt(max(V_d.mean(), 1e-4)))
            sigma_iv = implied_vol_newton(price, S_b, K_b, tau_b, r, q, otype_b, sigma_init=sigma_init)

            d_bench = bsm_delta(S_b, K_b, tau_b, r, q, sigma_iv, otype_b)
            d_bench_eq = call_equiv(d_bench, q, tau_b, otype_b)
            L_b = L_B(d_bench_eq)

            d_market_eq = call_equiv(delta_market, q, tau_b, otype_b)
            L_m = L_B(d_market_eq)

            R[:, day, mask] = (L_m - L_b)
            sigma_iv_panel[:, day, mask] = sigma_iv

    var_share_pc1 = np.empty(n_paths); var_share_pc2 = np.empty(n_paths); var_share_pc3 = np.empty(n_paths)
    same_sign_all = np.empty(n_paths); min_loading_all = np.empty(n_paths)
    pc1_all = np.empty((n_paths, T_full)); pc2_all = np.empty((n_paths, T_full)); pc3_all = np.empty((n_paths, T_full))

    for p in range(n_paths):
        Rp = R[p]
        Rp_c = Rp - Rp.mean(axis=0, keepdims=True)
        cov = Rp_c.T @ Rp_c / max(T_full - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
        u1, u2, u3 = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
        x1 = Rp_c @ u1; x2 = Rp_c @ u2; x3 = Rp_c @ u3
        if np.corrcoef(x1, np.abs(Rp).sum(axis=1))[0, 1] < 0:
            x1 = -x1; u1 = -u1
        pc1_all[p] = x1; pc2_all[p] = x2; pc3_all[p] = x3
        total = max(eigvals.sum(), 1e-30)
        var_share_pc1[p] = float(eigvals[0] / total)
        var_share_pc2[p] = float(eigvals[1] / total)
        var_share_pc3[p] = float(eigvals[2] / total)
        dom_sign = np.sign(u1[np.argmax(np.abs(u1))])
        same_sign_all[p] = float((np.sign(u1) == dom_sign).mean())
        min_loading_all[p] = float(np.abs(u1).min())

    adf_p1 = np.array([adf_p(x) for x in pc1_all]); kpss_p1 = np.array([kpss_p(x) for x in pc1_all])
    acf_p1 = np.array([acf1(x) for x in pc1_all])
    acf_p2 = np.array([acf1(x) for x in pc2_all]); adf_p2 = np.array([adf_p(x) for x in pc2_all])
    kpss_p2 = np.array([kpss_p(x) for x in pc2_all])
    acf_p3 = np.array([acf1(x) for x in pc3_all])
    is_i1 = (adf_p1 >= 0.05) & (kpss_p1 <= 0.05)

    return {
        "params": {"kappa": params.kappa, "theta": params.theta, "xi": params.xi,
                    "rho": params.rho, "v0": params.v0},
        "n_paths": int(n_paths), "T": int(T_full),
        "var_share_median": float(np.median(var_share_pc1)),
        "var_share_q25": float(np.quantile(var_share_pc1, 0.25)),
        "var_share_q75": float(np.quantile(var_share_pc1, 0.75)),
        "acf1_median": float(np.median(acf_p1)),
        "acf1_q25": float(np.quantile(acf_p1, 0.25)),
        "acf1_q75": float(np.quantile(acf_p1, 0.75)),
        "same_sign_rate_median": float(np.median(same_sign_all)),
        "same_sign_rate_q25": float(np.quantile(same_sign_all, 0.25)),
        "same_sign_rate_q75": float(np.quantile(same_sign_all, 0.75)),
        "min_abs_loading_median": float(np.median(min_loading_all)),
        "adf_p_median": float(np.nanmedian(adf_p1)),
        "kpss_p_median": float(np.nanmedian(kpss_p1)),
        "frac_i1": float(is_i1.mean()),
        "sigma_iv_mean": float(np.nanmean(sigma_iv_panel)),
        "sigma_iv_std_acrossopt": float(np.nanmean(np.nanstd(sigma_iv_panel, axis=2))),
        "var_share_pc2_median": float(np.median(var_share_pc2)),
        "acf1_pc2_median": float(np.median(acf_p2)),
        "adf_pc2_median": float(np.nanmedian(adf_p2)),
        "kpss_pc2_median": float(np.nanmedian(kpss_p2)),
        "var_share_pc3_median": float(np.median(var_share_pc3)),
        "acf1_pc3_median": float(np.median(acf_p3)),
    }

def main():
    base = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    configs = [
        ("base", base),
        ("low_kappa", HestonParams(kappa=1.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)),
        ("high_xi", HestonParams(kappa=2.0, theta=0.04, xi=0.8, rho=-0.7, v0=0.04)),
        ("strong_corr", HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.9, v0=0.04)),
        ("high_v0", HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.06)),
    ]
    results = {}
    for name, p in configs:
        print(f"\nHeston IV-matched {name}: {p}", flush=True)
        results[name] = run_heston_iv_matched(p, n_paths=40, T_total_days=348, seed=SEED)
        r = results[name]
        print(f"  var={r['var_share_median']:.3f}  acf={r['acf1_median']:.3f} "
              f"same_sign={r['same_sign_rate_median']:.3f} frac_i1={r['frac_i1']:.3f}",
              flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"configs": results, "seed": SEED,
                    "benchmark": "BSM-Delta at Heston sigma_IV (apples-to-apples with real)"},
                   f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
