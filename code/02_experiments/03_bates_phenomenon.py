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
    make_window_a_universe, bsm_delta, call_equiv, L_B, acf1, adf_p, kpss_p,
)
from src.synthetic.iv_match import implied_vol_newton
from src.synthetic.bates import (
    BatesParams, simulate_bates, cos_call_price_bates,
    bates_call_price_per_day, bates_delta_per_day_FD,
)

SEED = 2026
OUT_JSON = ROOT / "results" / "bates_phenomenon.json"

def run_bates_iv_matched(params: BatesParams, n_paths=40, T_total_days=348,
                          S0=100.0, r=0.04, q=0.0117, seed=SEED):
    rng = np.random.default_rng(seed)
    K_arr, T_exp_arr, type_arr = make_window_a_universe(S0)
    N = len(K_arr)

    print(f"  Sim Bates: {n_paths}x{T_total_days}", flush=True)
    S, V = simulate_bates(params, S0, r, q, T_total_days, n_paths, rng)
    T_full = T_total_days + 1

    R = np.zeros((n_paths, T_full, N), dtype=np.float64)

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

            price = bates_call_price_per_day(S_d, V_d, K_sub, tau, type_sub, r, q, params)
            delta_market = bates_delta_per_day_FD(S_d, V_d, K_sub, tau, type_sub, r, q, params)

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
            R[:, day, mask] = L_m - L_b

    var_share_pc1 = np.empty(n_paths)
    var_share_pc2 = np.empty(n_paths)
    var_share_pc3 = np.empty(n_paths)
    same_sign_pc1 = np.empty(n_paths)
    min_loading_pc1 = np.empty(n_paths)
    pc1_all = np.empty((n_paths, T_full))
    pc2_all = np.empty((n_paths, T_full))
    pc3_all = np.empty((n_paths, T_full))

    for p in range(n_paths):
        Rp = R[p]
        Rp_c = Rp - Rp.mean(axis=0, keepdims=True)
        cov = Rp_c.T @ Rp_c / max(T_full - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        u1, u2, u3 = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
        x1 = Rp_c @ u1; x2 = Rp_c @ u2; x3 = Rp_c @ u3
        if np.corrcoef(x1, np.abs(Rp).sum(axis=1))[0, 1] < 0:
            x1 = -x1; u1 = -u1
        pc1_all[p] = x1; pc2_all[p] = x2; pc3_all[p] = x3
        total = max(eigvals.sum(), 1e-30)
        var_share_pc1[p] = float(eigvals[0] / total)
        var_share_pc2[p] = float(eigvals[1] / total)
        var_share_pc3[p] = float(eigvals[2] / total)
        dom = np.sign(u1[np.argmax(np.abs(u1))])
        same_sign_pc1[p] = float((np.sign(u1) == dom).mean())
        min_loading_pc1[p] = float(np.abs(u1).min())

    def acf_arr(M):
        return np.array([acf1(x) for x in M])

    def adf_arr(M):
        return np.array([adf_p(x) for x in M])

    def kpss_arr(M):
        return np.array([kpss_p(x) for x in M])

    adf_p1 = adf_arr(pc1_all); kpss_p1 = kpss_arr(pc1_all); acf_p1 = acf_arr(pc1_all)
    adf_p2 = adf_arr(pc2_all); kpss_p2 = kpss_arr(pc2_all); acf_p2 = acf_arr(pc2_all)
    adf_p3 = adf_arr(pc3_all); kpss_p3 = kpss_arr(pc3_all); acf_p3 = acf_arr(pc3_all)
    is_i1 = (adf_p1 >= 0.05) & (kpss_p1 <= 0.05)

    return {
        "params": {"kappa": params.kappa, "theta": params.theta, "xi": params.xi,
                    "rho": params.rho, "v0": params.v0,
                    "lam": params.lam, "mu_J": params.mu_J, "delta_J": params.delta_J,
                    "jump_compensator": params.jump_compensator},
        "n_paths": int(n_paths), "T": int(T_full),

        "var_share_pc1_median": float(np.median(var_share_pc1)),
        "acf1_pc1_median": float(np.median(acf_p1)),
        "adf_pc1_median": float(np.nanmedian(adf_p1)),
        "kpss_pc1_median": float(np.nanmedian(kpss_p1)),
        "frac_pc1_i1": float(is_i1.mean()),
        "same_sign_pc1": float(np.median(same_sign_pc1)),
        "min_loading_pc1": float(np.median(min_loading_pc1)),

        "var_share_pc2_median": float(np.median(var_share_pc2)),
        "acf1_pc2_median": float(np.median(acf_p2)),
        "adf_pc2_median": float(np.nanmedian(adf_p2)),
        "kpss_pc2_median": float(np.nanmedian(kpss_p2)),

        "var_share_pc3_median": float(np.median(var_share_pc3)),
        "acf1_pc3_median": float(np.median(acf_p3)),
    }

def main():
    base = BatesParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04,
                        lam=2.0, mu_J=-0.05, delta_J=0.10)
    configs = [
        ("base", base),
        ("low_lam", BatesParams(**{**base.__dict__, "lam": 0.5})),
        ("high_lam", BatesParams(**{**base.__dict__, "lam": 5.0})),
        ("large_jump", BatesParams(**{**base.__dict__, "delta_J": 0.20, "mu_J": -0.10})),
        ("symm_jump", BatesParams(**{**base.__dict__, "mu_J": 0.0})),
    ]
    results = {}
    for name, p in configs:
        print(f"\n[B8a] Bates {name}: lam={p.lam}, mu_J={p.mu_J}, delta_J={p.delta_J}", flush=True)
        results[name] = run_bates_iv_matched(p, n_paths=40, T_total_days=348, seed=SEED)
        r = results[name]
        print(f"  PC1 var={r['var_share_pc1_median']:.3f}  acf={r['acf1_pc1_median']:.3f}  frac_i1={r['frac_pc1_i1']:.3f}",
              flush=True)
        print(f"  PC2 var={r['var_share_pc2_median']:.3f}  acf={r['acf1_pc2_median']:.3f}",
              flush=True)
        print(f"  PC3 var={r['var_share_pc3_median']:.3f}  acf={r['acf1_pc3_median']:.3f}",
              flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"configs": results, "seed": SEED,
                    "benchmark": "BSM-Delta at Bates sigma_IV (analytic COS, no MC noise)"},
                   f, indent=2)
    print(f"\n[B8a] Wrote {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
