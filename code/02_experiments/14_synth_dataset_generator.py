from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.synthetic.heston import (
    HestonParams, simulate_heston_qe, cos_call_price, bsm_delta, call_equiv, L_B,
    make_window_a_universe,
)
from src.synthetic.iv_match import (
    bsm_price_call, bsm_vega, implied_vol_newton,
    heston_price_per_day, heston_delta_per_day_FD,
)
from src.synthetic.bates import (
    BatesParams, simulate_bates,
    bates_call_price_per_day, bates_delta_per_day_FD,
)

SEED = 2026
N_PATHS = 200
T_DAYS = 348
S0 = 100.0
R_RATE = 0.04
Q_RATE = 0.0117
THRESH_QUANTILE = 0.90

OUT_HESTON = ROOT / "data" / "synthetic" / "heston" / "heston_panel.npz"
OUT_BATES  = ROOT / "data" / "synthetic" / "bates"  / "bates_panel.npz"

def generate_panel(model: str, params, K_arr, T_exp_arr, type_arr, seed,
                    n_paths=N_PATHS, T_total_days=T_DAYS,
                    S0=S0, r=R_RATE, q=Q_RATE):
    rng = np.random.default_rng(seed)
    N_opt = len(K_arr)

    if model == "heston":
        S, V = simulate_heston_qe(params, S0, r, q, T_total_days, n_paths, rng)
        price_fn = heston_price_per_day
        delta_fn = heston_delta_per_day_FD
    elif model == "bates":
        S, V = simulate_bates(params, S0, r, q, T_total_days, n_paths, rng)
        price_fn = bates_call_price_per_day
        delta_fn = bates_delta_per_day_FD
    else:
        raise ValueError(model)

    T_full = T_total_days + 1
    R = np.zeros((n_paths, T_full, N_opt), dtype=np.float64)
    delta_market_panel = np.zeros((n_paths, T_full, N_opt), dtype=np.float64)
    sigma_iv_panel = np.zeros((n_paths, T_full, N_opt), dtype=np.float64)
    moneyness_panel = np.zeros((n_paths, T_full, N_opt), dtype=np.float64)
    tau_panel = np.zeros((n_paths, T_full, N_opt), dtype=np.float64)

    t_start = time.time()
    for day in range(T_full):
        if day % 50 == 0:
            print(f"    {model} day {day}/{T_full - 1}  ({time.time() - t_start:.0f}s)", flush=True)
        S_d = S[:, day]; V_d = V[:, day]
        days_year = day / 365.25
        for T_uniq in np.unique(T_exp_arr):
            mask = T_exp_arr == T_uniq
            tau = float(T_uniq - days_year)
            tau = max(tau, 1.0 / 365.0)
            K_sub = K_arr[mask]; type_sub = type_arr[mask]

            price = price_fn(S_d, V_d, K_sub, tau, type_sub, r, q, params)
            delta_m = delta_fn(S_d, V_d, K_sub, tau, type_sub, r, q, params)
            S_b = np.broadcast_to(S_d[:, None], price.shape)
            K_b = np.broadcast_to(K_sub[None, :], price.shape)
            tau_b = np.broadcast_to(tau, price.shape)
            otype_b = np.broadcast_to(type_sub[None, :], price.shape)
            sigma_init = float(np.sqrt(max(V_d.mean(), 1e-4)))
            sigma_iv = implied_vol_newton(price, S_b, K_b, tau_b, r, q, otype_b,
                                           sigma_init=sigma_init)

            d_bench = bsm_delta(S_b, K_b, tau_b, r, q, sigma_iv, otype_b)
            d_bench_eq = call_equiv(d_bench, q, tau_b, otype_b)
            d_market_eq = call_equiv(delta_m, q, tau_b, otype_b)

            R[:, day, mask] = L_B(d_market_eq) - L_B(d_bench_eq)
            delta_market_panel[:, day, mask] = d_market_eq
            sigma_iv_panel[:, day, mask] = sigma_iv
            moneyness_panel[:, day, mask] = np.log(K_b / S_b)
            tau_panel[:, day, mask] = tau

    pc1_all = np.empty((n_paths, T_full), dtype=np.float64)
    for p in range(n_paths):
        Rp = R[p]
        Rp_c = Rp - Rp.mean(axis=0, keepdims=True)
        cov = Rp_c.T @ Rp_c / max(T_full - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u1 = eigvecs[:, -1]
        x = Rp_c @ u1
        if np.corrcoef(x, np.abs(Rp).sum(axis=1))[0, 1] < 0:
            x = -x
        pc1_all[p] = x

    dpc1_all = np.diff(pc1_all, axis=1)
    abs_dpc1 = np.abs(dpc1_all)

    rng_split = np.random.default_rng(seed)
    perm = rng_split.permutation(n_paths)
    n_train = int(n_paths * 0.70)
    n_val = int(n_paths * 0.15)
    train_paths = np.sort(perm[:n_train])
    val_paths = np.sort(perm[n_train:n_train + n_val])
    test_paths = np.sort(perm[n_train + n_val:])

    train_abs = abs_dpc1[train_paths].ravel()
    threshold = float(np.quantile(train_abs, THRESH_QUANTILE))
    labels = (abs_dpc1 > threshold).astype(np.int8)

    return {
        "R": R.astype(np.float32),
        "delta_market_eq": delta_market_panel.astype(np.float32),
        "sigma_iv": sigma_iv_panel.astype(np.float32),
        "moneyness": moneyness_panel.astype(np.float32),
        "tau": tau_panel.astype(np.float32),
        "S": S.astype(np.float32),
        "V": V.astype(np.float32),
        "pc1": pc1_all.astype(np.float32),
        "dpc1": dpc1_all.astype(np.float32),
        "labels": labels,
        "threshold": np.float32(threshold),
        "train_paths": train_paths.astype(np.int32),
        "val_paths": val_paths.astype(np.int32),
        "test_paths": test_paths.astype(np.int32),
        "K": K_arr.astype(np.float32),
        "T_expiry": T_exp_arr.astype(np.float32),
        "option_type": np.asarray(type_arr, dtype="U1"),
        "params": np.array([params.__dict__], dtype=object),
        "seed": np.int64(seed),
    }

def main():
    K_arr, T_exp_arr, type_arr = make_window_a_universe(S0)
    print(f"Universe: N_options = {len(K_arr)}, T_expiry unique = {sorted(set(T_exp_arr))}",
          flush=True)

    print("\n=== Heston ===", flush=True)
    h_params = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    print(f"params: {h_params}", flush=True)
    OUT_HESTON.parent.mkdir(parents=True, exist_ok=True)
    h_data = generate_panel("heston", h_params, K_arr, T_exp_arr, type_arr, seed=SEED)
    print(f"Heston panel: PC1 shape {h_data['pc1'].shape}, "
          f"label rate {h_data['labels'].mean():.3f}, threshold {h_data['threshold']:.5f}", flush=True)
    np.savez_compressed(OUT_HESTON, **h_data)
    size_mb = OUT_HESTON.stat().st_size / 1e6
    print(f"Wrote {OUT_HESTON} ({size_mb:.1f} MB)", flush=True)

    print("\n=== Bates ===", flush=True)
    b_params = BatesParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04,
                            lam=2.0, mu_J=-0.05, delta_J=0.10)
    print(f"params: {b_params}", flush=True)
    OUT_BATES.parent.mkdir(parents=True, exist_ok=True)
    b_data = generate_panel("bates", b_params, K_arr, T_exp_arr, type_arr, seed=SEED)
    print(f"Bates panel: PC1 shape {b_data['pc1'].shape}, "
          f"label rate {b_data['labels'].mean():.3f}, threshold {b_data['threshold']:.5f}", flush=True)
    np.savez_compressed(OUT_BATES, **b_data)
    size_mb = OUT_BATES.stat().st_size / 1e6
    print(f"Wrote {OUT_BATES} ({size_mb:.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
