from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_RATIOS = np.array([5500, 6100, 6500, 6800, 7100, 7500, 8200], dtype=np.float64) / 5158.86
_T_EXPIRY_YEARS = (2.0, 2.5)

def make_window_a_universe(S0: float = 100.0):
    K_list, T_list, type_list = [], [], []
    for T_exp in _T_EXPIRY_YEARS:
        for ratio in _RATIOS:
            K_list.append(S0 * ratio); T_list.append(T_exp); type_list.append("C")
        for ratio in _RATIOS:
            if T_exp == _T_EXPIRY_YEARS[1] and ratio == _RATIOS[-1]:
                continue
            K_list.append(S0 * ratio); T_list.append(T_exp); type_list.append("P")
    return (np.array(K_list, dtype=np.float64),
            np.array(T_list, dtype=np.float64),
            np.array(type_list, dtype="U1"))

@dataclass
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

def simulate_heston_qe(params: HestonParams, S0, r, q, T_total_days, n_paths, rng):
    T = T_total_days
    dt = (T / 365.25) / T
    psi_c = 1.5
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho

    e = np.exp(-kappa * dt)
    K0 = -rho * kappa * theta / xi * dt
    K1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = 0.5 * dt * (1 - rho ** 2)
    K4 = K3

    V = np.empty((n_paths, T + 1), dtype=np.float64)
    S = np.empty((n_paths, T + 1), dtype=np.float64)
    V[:, 0] = params.v0
    S[:, 0] = S0

    for t in range(T):
        v_curr = V[:, t]
        m = theta + (v_curr - theta) * e
        s2 = (v_curr * xi ** 2 * e / kappa) * (1 - e) + 0.5 * theta * xi ** 2 / kappa * (1 - e) ** 2
        psi = s2 / (m ** 2 + 1e-300)

        b2 = 2.0 / np.maximum(psi, 1e-12) - 1 + np.sqrt(2.0 / np.maximum(psi, 1e-12)) \
             * np.sqrt(np.maximum(2.0 / np.maximum(psi, 1e-12) - 1, 0.0))
        b2 = np.maximum(b2, 1e-12)
        a = m / (1 + b2)
        Z = rng.standard_normal(n_paths)
        v_next_A = a * (np.sqrt(b2) + Z) ** 2

        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / np.maximum(m, 1e-300)
        U = rng.uniform(size=n_paths)
        v_next_B = np.where(U <= p, 0.0, np.log((1 - p) / np.maximum(1 - U, 1e-300)) / np.maximum(beta, 1e-300))

        v_next = np.where(psi <= psi_c, v_next_A, v_next_B)
        v_next = np.maximum(v_next, 0.0)
        V[:, t + 1] = v_next

        Z_S = rng.standard_normal(n_paths)
        log_S_next = (np.log(S[:, t]) + (r - q) * dt + K0
                      + K1 * v_curr + K2 * v_next
                      + np.sqrt(np.maximum(K3 * v_curr + K4 * v_next, 0.0)) * Z_S)
        S[:, t + 1] = np.exp(log_S_next)
    return S, V

def heston_cf(u, S, v0, tau, r, q, params: HestonParams):
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + u ** 2))
    g2 = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    exp_dt = np.exp(-d * tau)
    C = (r - q) * iu * tau + kappa * theta / xi ** 2 * (
        (kappa - rho * xi * iu - d) * tau - 2.0 * np.log((1 - g2 * exp_dt) / (1 - g2 + 1e-300)))
    D = (kappa - rho * xi * iu - d) / xi ** 2 * (1 - exp_dt) / (1 - g2 * exp_dt + 1e-300)
    return np.exp(C + D * v0 + iu * np.log(S))

def cos_call_price(S, K, tau, r, q, v0, params: HestonParams, N_cos: int = 256, L: float = 12.0):
    S = np.atleast_1d(np.asarray(S, dtype=np.float64))
    v0 = np.atleast_1d(np.asarray(v0, dtype=np.float64))
    K = np.atleast_1d(np.asarray(K, dtype=np.float64))
    tau = float(tau)

    P = S.shape[0]
    Nk = K.shape[0]

    sigma_eff = np.sqrt(np.maximum(v0.mean(), 1e-8))
    width = L * sigma_eff * np.sqrt(tau) + 5 * sigma_eff * np.sqrt(tau)

    log_S_over_K = np.log(S[:, None] / K[None, :])

    a_global = log_S_over_K.min() - width
    b_global = log_S_over_K.max() + width
    a, b = float(a_global), float(b_global)

    k_idx = np.arange(N_cos)
    u = k_idx * np.pi / (b - a)

    cf_vals = np.empty((P, N_cos), dtype=np.complex128)
    for j in range(P):
        cf_vals[j] = heston_cf(u, S[j], v0[j], tau, r, q, params)

    c, d = 0.0, b
    arg_d = k_idx * np.pi * (d - a) / (b - a)
    arg_c = k_idx * np.pi * (c - a) / (b - a)
    omega = k_idx * np.pi / (b - a)
    denom = 1.0 + omega ** 2
    chi = (np.cos(arg_d) * np.exp(d) - np.cos(arg_c) * np.exp(c)
           + omega * np.sin(arg_d) * np.exp(d) - omega * np.sin(arg_c) * np.exp(c)) / denom
    psi = np.where(k_idx == 0, d - c, (np.sin(arg_d) - np.sin(arg_c)) / np.maximum(omega, 1e-300))
    V_k_perK = (2.0 / (b - a)) * (chi - psi)

    log_K = np.log(K)
    exp_term = np.exp(-1j * u[None, :] * (log_K[:, None] + a))
    integrand = cf_vals[:, None, :] * exp_term[None, :, :]
    series = np.real(integrand) * V_k_perK[None, None, :]
    series[:, :, 0] *= 0.5
    sum_per_pair = series.sum(axis=2)
    price = K[None, :] * np.exp(-r * tau) * sum_per_pair
    return price

def heston_call_price_per_day(S, V, K, tau, r, q, params, h_rel=0.01):
    return cos_call_price(S, K, tau, r, q, V, params)

def heston_delta_per_day(S, V, K, tau, otype, r, q, params, h_rel=0.01):
    h = h_rel * S
    P_plus = cos_call_price(S + h, K, tau, r, q, V, params)
    P_minus = cos_call_price(S - h, K, tau, r, q, V, params)
    delta_call = (P_plus - P_minus) / (2.0 * h[:, None])
    is_call = (otype == "C")
    delta_put = delta_call - np.exp(-q * tau)
    return np.where(is_call[None, :], delta_call, delta_put)

def bsm_delta(S, K, tau, r, q, sigma, otype):
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-300))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    delta_call = np.exp(-q * tau) * norm.cdf(d1)
    is_call = otype == "C"
    return np.where(is_call, delta_call, delta_call - np.exp(-q * tau))

def call_equiv(delta, q, tau, otype):
    is_call = otype == "C"
    return np.where(is_call, delta, delta + np.exp(-q * tau))

def L_B(delta_eq):
    return (2.0 * delta_eq - 1.0) ** 2

def acf1(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = (x * x).sum()
    return float((x[:-1] * x[1:]).sum() / max(denom, 1e-30))

def adf_p(x):
    from statsmodels.tsa.stattools import adfuller
    try:
        return float(adfuller(x, autolag="AIC", regression="c")[1])
    except Exception:
        return float("nan")

def kpss_p(x):
    from statsmodels.tsa.stattools import kpss
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, *_ = kpss(x, regression="c", nlags="auto")
        return float(p)
    except Exception:
        return float("nan")
