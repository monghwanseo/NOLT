from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from src.synthetic.heston import (
    HestonParams, simulate_heston_qe, cos_call_price, make_window_a_universe,
)
from src.synthetic.iv_match import (
    bsm_price_call, bsm_vega, implied_vol_newton, heston_delta_per_day_FD,
    heston_price_per_day,
)
sys.path.insert(0, str(ROOT / "code"))
from src.metrics.linearity import L_B
from src.metrics.bsm_greeks import bsm_delta, bsm_call_equivalent_delta

SEED = 2026
R_RATE = 0.04
Q_RATE = 0.0117
S0 = 100.0
T_TOTAL_DAYS = 348
N_PATHS = 1
HESTON_BASE = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)

def call_equiv(delta_raw, q, tau, otype):
    discount = np.exp(q * tau)
    is_call = (otype == "C")
    return np.where(is_call, discount * delta_raw, 1.0 + discount * delta_raw)

def build_residual_matrix_one_path(params=HESTON_BASE, T_total_days=T_TOTAL_DAYS,
                                    seed=SEED):
    rng = np.random.default_rng(seed)
    K_arr, T_exp_arr, type_arr = make_window_a_universe(S0)
    N = len(K_arr)

    S, V = simulate_heston_qe(params, S0, R_RATE, Q_RATE, T_total_days, N_PATHS, rng)
    T_full = T_total_days + 1
    R = np.zeros((T_full, N), dtype=np.float64)

    for day in range(T_full):
        S_d = S[:, day]
        V_d = V[:, day]
        elapsed_year = day / 365.25
        for T_uniq in np.unique(T_exp_arr):
            mask = T_exp_arr == T_uniq
            tau = float(T_uniq - elapsed_year)
            tau = max(tau, 1.0 / 365.0)
            K_sub = K_arr[mask]
            type_sub = type_arr[mask]

            price = heston_price_per_day(S_d, V_d, K_sub, tau, type_sub,
                                          R_RATE, Q_RATE, params)
            delta_mkt = heston_delta_per_day_FD(S_d, V_d, K_sub, tau, type_sub,
                                                  R_RATE, Q_RATE, params)

            S_b = np.broadcast_to(S_d[:, None], price.shape)
            K_b = np.broadcast_to(K_sub[None, :], price.shape)
            tau_b = np.broadcast_to(tau, price.shape)
            otype_b = np.broadcast_to(type_sub[None, :], price.shape)
            sigma_init = float(np.sqrt(max(V_d.mean(), 1e-4)))
            sigma_iv = implied_vol_newton(price, S_b, K_b, tau_b, R_RATE, Q_RATE,
                                            otype_b, sigma_init=sigma_init)

            d_bench = bsm_delta(S_b, K_b, tau_b, R_RATE, Q_RATE, sigma_iv, otype_b)
            d_bench_eq = call_equiv(d_bench, Q_RATE, tau_b, otype_b)
            L_bench = L_B(d_bench_eq)

            d_mkt_eq = call_equiv(delta_mkt, Q_RATE, tau_b, otype_b)
            L_mkt = L_B(d_mkt_eq)

            R[day, mask] = (L_mkt - L_bench)[0]

    return R, V[0], K_arr, T_exp_arr, type_arr

def compute_w_theoretical(K_arr, T_exp_arr, type_arr, params=HESTON_BASE):
    from scipy.stats import norm
    sigma_0 = np.sqrt(params.theta)
    w = np.zeros_like(K_arr, dtype=np.float64)
    delta_eq_base = np.zeros_like(K_arr, dtype=np.float64)
    for i in range(len(K_arr)):
        K = K_arr[i]
        tau = T_exp_arr[i]
        sqrt_t = np.sqrt(tau)
        d1 = (np.log(S0 / K) + (R_RATE - Q_RATE + 0.5 * sigma_0 ** 2) * tau) / (sigma_0 * sqrt_t)
        d2 = d1 - sigma_0 * sqrt_t
        phi_d1 = norm.pdf(d1)
        N_d1 = norm.cdf(d1)
        discount = np.exp(Q_RATE * tau)

        if type_arr[i] == "C":
            delta_eq_base[i] = discount * (np.exp(-Q_RATE * tau) * N_d1)
            ddelta_dsigma = np.exp(-Q_RATE * tau) * phi_d1 * (-d2 / sigma_0)
            ddelta_eq_dsigma = discount * ddelta_dsigma
        else:

            delta_eq_base[i] = N_d1
            ddelta_dsigma = np.exp(-Q_RATE * tau) * phi_d1 * (-d2 / sigma_0)
            ddelta_eq_dsigma = discount * ddelta_dsigma * np.exp(-Q_RATE * tau)

            ddelta_eq_dsigma = phi_d1 * (-d2 / sigma_0)
        w[i] = 4.0 * (2.0 * delta_eq_base[i] - 1.0) * ddelta_eq_dsigma
    return w

def main():
    print("=" * 78)
    print("E2 — Heston rank-1 separability numerical check")
    print(f"params: kappa={HESTON_BASE.kappa}, theta={HESTON_BASE.theta}, "
          f"xi={HESTON_BASE.xi}, rho={HESTON_BASE.rho}, v0={HESTON_BASE.v0}")
    print(f"T_total={T_TOTAL_DAYS} days, seed={SEED}, 1 path")
    print("=" * 78)

    R, V, K_arr, T_exp_arr, type_arr = build_residual_matrix_one_path()
    T_full, N = R.shape
    print(f"  Residual matrix shape: ({T_full}, {N})")
    print(f"  v_t mean = {V.mean():.5f}, std = {V.std():.5f}")

    psi = HESTON_BASE.rho * HESTON_BASE.xi * np.sqrt(np.maximum(V, 0.0) / HESTON_BASE.theta)
    print(f"  psi(v_t) mean = {psi.mean():.5f}, std = {psi.std():.5f}")

    Rc = R - R.mean(axis=0, keepdims=True)
    U_svd, S_svd, Vt_svd = np.linalg.svd(Rc, full_matrices=False)
    u1 = U_svd[:, 0]
    v1 = Vt_svd[0, :]
    sigma1 = S_svd[0]

    if np.corrcoef(u1, np.abs(R).sum(axis=1))[0, 1] < 0:
        u1 = -u1
        v1 = -v1

    var_shares = (S_svd ** 2) / max((S_svd ** 2).sum(), 1e-30)
    print(f"  SVD singular values (top 5): {S_svd[:5]}")
    print(f"  Variance share (top 3): {var_shares[:3]}")

    pearson_u1_psi = float(np.corrcoef(u1, psi)[0, 1])

    w_th = compute_w_theoretical(K_arr, T_exp_arr, type_arr)

    cos_v1_wth_raw = float(np.dot(v1, w_th) / (np.linalg.norm(v1) * np.linalg.norm(w_th) + 1e-30))
    cos_v1_wth_abs = abs(cos_v1_wth_raw)

    R_rank1 = sigma1 * np.outer(u1, v1) if np.corrcoef(U_svd[:, 0], u1)[0, 1] > 0 \
        else -sigma1 * np.outer(U_svd[:, 0], Vt_svd[0, :])

    R_rank1_centered = S_svd[0] * np.outer(U_svd[:, 0], Vt_svd[0, :])
    rec_err = float(np.linalg.norm(Rc - R_rank1_centered) / max(np.linalg.norm(Rc), 1e-30))

    psi_c = psi - psi.mean()
    if psi_c.std() > 1e-12:
        psi_norm = psi_c / np.linalg.norm(psi_c)

        v_star = Rc.T @ psi_norm
        recon_psi_v = np.outer(psi_norm, v_star)
        rec_err_psi = float(np.linalg.norm(Rc - recon_psi_v) / max(np.linalg.norm(Rc), 1e-30))
        cos_vstar_wth = float(np.dot(v_star, w_th) / (np.linalg.norm(v_star) * np.linalg.norm(w_th) + 1e-30))
    else:
        rec_err_psi = float("nan")
        cos_vstar_wth = float("nan")

    print()
    print(f"  Pearson(u1_SVD, psi(v_t))          = {pearson_u1_psi:+.4f}")
    print(f"  Cos_sim(v1_SVD, w_theoretical)     = {cos_v1_wth_raw:+.4f}  (|.|={cos_v1_wth_abs:.4f})")
    print(f"  ||R - sigma_1 u_1 v_1^T||_F / ||R||_F (rank-1 reconstruct) = {rec_err:.4f}")
    print(f"  ||R - psi v*^T||_F / ||R||_F  (psi-constrained rank-1)     = {rec_err_psi:.4f}")
    print(f"  Cos_sim(v*_psi, w_theoretical)     = {cos_vstar_wth:+.4f}")

    out = {
        "params": {"kappa": HESTON_BASE.kappa, "theta": HESTON_BASE.theta,
                    "xi": HESTON_BASE.xi, "rho": HESTON_BASE.rho, "v0": HESTON_BASE.v0},
        "T_total_days": T_TOTAL_DAYS,
        "N_options": int(N),
        "seed": SEED,
        "v_t_mean": float(V.mean()), "v_t_std": float(V.std()),
        "psi_mean": float(psi.mean()), "psi_std": float(psi.std()),
        "svd_singular_top5": S_svd[:5].tolist(),
        "var_share_top3": var_shares[:3].tolist(),
        "pearson_u1_psi": pearson_u1_psi,
        "cos_v1_w_theoretical": cos_v1_wth_raw,
        "cos_v1_w_theoretical_abs": cos_v1_wth_abs,
        "rank1_recon_error_F": rec_err,
        "psi_constrained_recon_error_F": rec_err_psi,
        "cos_vstar_w_theoretical": cos_vstar_wth,
    }
    out_json = ROOT / "results" / "E2_heston_rank1.json"
    out_json.write_text(json.dumps(out, indent=2,
                                     default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_json}")

    df = pd.DataFrame([{
        "Quantity": "Pearson(u1_SVD, psi(v_t))",
        "Value": pearson_u1_psi,
        "Notes": "Time-mode identification: driver = rho*xi*sqrt(v_t/theta)",
    }, {
        "Quantity": "Cos_sim(v1_SVD, w_theoretical)",
        "Value": cos_v1_wth_raw,
        "Notes": "Cross-section identification: 1st-order Taylor of L_B in sigma",
    }, {
        "Quantity": "Rank-1 reconstruction error ||R-s1*u1*v1^T||_F / ||R||_F",
        "Value": rec_err,
        "Notes": "Best rank-1 SVD approximation quality (lower is better)",
    }, {
        "Quantity": "psi-constrained rank-1 reconstruction error",
        "Value": rec_err_psi,
        "Notes": "Best rank-1 with time-mode fixed to psi(v_t)",
    }, {
        "Quantity": "Variance share captured by sigma_1^2",
        "Value": float(var_shares[0]),
        "Notes": "Fraction of total cross-section variance in leading mode",
    }])
    out_csv = ROOT / "paper" / "tables" / "T_heston_rank1.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"saved: {out_csv}")

if __name__ == "__main__":
    main()
