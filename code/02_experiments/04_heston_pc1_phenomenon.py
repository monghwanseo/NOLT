from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SEED = 2026
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.synthetic.heston import (
    HestonParams, simulate_heston_qe, heston_delta_per_day,
    bsm_delta, call_equiv, L_B,
    make_window_a_universe, acf1, adf_p, kpss_p,
)

OUT_JSON = ROOT / "results" / "heston_pc1_phenomenon.json"

def run_heston_config(params: HestonParams, n_paths=40, T_total_days=348,
                      S0=100.0, r=0.04, q=0.0117, seed=2026):
    rng = np.random.default_rng(seed)
    K_arr, T_exp_arr, type_arr = make_window_a_universe(S0)
    N = len(K_arr)

    print(f"  Sim Heston QE: {n_paths} paths x {T_total_days} days...", flush=True)
    S, V = simulate_heston_qe(params, S0, r, q, T_total_days, n_paths, rng)
    T_full = T_total_days + 1

    sigma_const = float(np.sqrt(params.v0))

    R = np.zeros((n_paths, T_full, N), dtype=np.float64)

    print(f"  COS pricer: {T_full} days x {N} options...", flush=True)
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
            delta_market = heston_delta_per_day(S_d, V_d, K_sub, tau, type_sub, r, q, params)
            tau_b = np.broadcast_to(tau, delta_market.shape)
            otype_b = np.broadcast_to(type_sub[None, :], delta_market.shape)
            delta_market_eq = call_equiv(delta_market, q, tau_b, otype_b)
            L_market = L_B(delta_market_eq)

            S_b = S_d[:, None]
            K_b = K_sub[None, :]
            d_bench = bsm_delta(S_b, K_b, tau, r, q, sigma_const, otype_b)
            d_bench_eq = call_equiv(d_bench, q, tau_b, otype_b)
            L_bench = L_B(d_bench_eq)
            R[:, day, mask] = (L_market - L_bench)

    pc1_all = np.empty((n_paths, T_full), dtype=np.float64)
    var_share_all = np.empty(n_paths)
    same_sign_rate_all = np.empty(n_paths)
    min_abs_loading_all = np.empty(n_paths)

    for p in range(n_paths):
        Rp = R[p]
        Rp_c = Rp - Rp.mean(axis=0, keepdims=True)
        cov = Rp_c.T @ Rp_c / max(T_full - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u1 = eigvecs[:, -1]
        pc1_p = Rp_c @ u1

        if np.corrcoef(pc1_p, np.abs(Rp).sum(axis=1))[0, 1] < 0:
            pc1_p = -pc1_p
            u1 = -u1
        pc1_all[p] = pc1_p
        var_share_all[p] = float(eigvals[-1] / max(eigvals.sum(), 1e-30))

        dom_sign = np.sign(u1[np.argmax(np.abs(u1))])
        same_sign_rate_all[p] = float((np.sign(u1) == dom_sign).mean())
        min_abs_loading_all[p] = float(np.abs(u1).min())

    adf_pvals = np.array([adf_p(x) for x in pc1_all])
    kpss_pvals = np.array([kpss_p(x) for x in pc1_all])
    acf_pc1 = np.array([acf1(x) for x in pc1_all])
    is_i1 = (adf_pvals >= 0.05) & (kpss_pvals <= 0.05)

    return {
        "params": {"kappa": params.kappa, "theta": params.theta, "xi": params.xi,
                    "rho": params.rho, "v0": params.v0},
        "n_paths": int(n_paths),
        "T": int(T_full),
        "var_share_median": float(np.median(var_share_all)),
        "var_share_q25": float(np.quantile(var_share_all, 0.25)),
        "var_share_q75": float(np.quantile(var_share_all, 0.75)),
        "acf1_median": float(np.median(acf_pc1)),
        "acf1_q25": float(np.quantile(acf_pc1, 0.25)),
        "acf1_q75": float(np.quantile(acf_pc1, 0.75)),
        "same_sign_rate_median": float(np.median(same_sign_rate_all)),
        "same_sign_rate_q25": float(np.quantile(same_sign_rate_all, 0.25)),
        "same_sign_rate_q75": float(np.quantile(same_sign_rate_all, 0.75)),
        "min_abs_loading_median": float(np.median(min_abs_loading_all)),
        "adf_p_median": float(np.nanmedian(adf_pvals)),
        "kpss_p_median": float(np.nanmedian(kpss_pvals)),
        "frac_i1": float(is_i1.mean()),
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
        print(f"\nRunning Heston {name}: {p}", flush=True)
        results[name] = run_heston_config(p, n_paths=40, T_total_days=348, seed=SEED)
        print(f"  var_share_median = {results[name]['var_share_median']:.3f}", flush=True)
        print(f"  acf1_median = {results[name]['acf1_median']:.3f}", flush=True)
        print(f"  same_sign_median = {results[name]['same_sign_rate_median']:.3f}", flush=True)
        print(f"  frac_i1 = {results[name]['frac_i1']:.3f}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"configs": results, "seed": SEED}, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
