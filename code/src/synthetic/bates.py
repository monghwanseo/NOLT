from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass
class BatesParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    lam: float
    mu_J: float
    delta_J: float

    @property
    def jump_compensator(self) -> float:
        return float(np.exp(self.mu_J + 0.5 * self.delta_J ** 2) - 1.0)

def simulate_bates(params: BatesParams, S0: float, r: float, q: float,
                    T_total_days: int, n_paths: int, rng: np.random.Generator):
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

    kappa_J = params.jump_compensator

    for t in range(T):
        v_curr = V[:, t]
        m = theta + (v_curr - theta) * e
        s2 = (v_curr * xi ** 2 * e / kappa) * (1 - e) + 0.5 * theta * xi ** 2 / kappa * (1 - e) ** 2
        psi = s2 / (m ** 2 + 1e-300)

        b2_arg = np.maximum(2.0 / np.maximum(psi, 1e-12) - 1, 0.0)
        b2 = 2.0 / np.maximum(psi, 1e-12) - 1 + np.sqrt(2.0 / np.maximum(psi, 1e-12)) * np.sqrt(b2_arg)
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

        N_jumps = rng.poisson(params.lam * dt, size=n_paths)
        Z_J = rng.standard_normal(n_paths)
        jump_total = params.mu_J * N_jumps + params.delta_J * np.sqrt(np.maximum(N_jumps, 0)) * Z_J

        Z_S = rng.standard_normal(n_paths)
        diffusion = K0 + K1 * v_curr + K2 * v_next \
                     + np.sqrt(np.maximum(K3 * v_curr + K4 * v_next, 0.0)) * Z_S

        log_S_next = np.log(S[:, t]) + (r - q) * dt - params.lam * kappa_J * dt + diffusion + jump_total
        S[:, t + 1] = np.exp(log_S_next)

    return S, V

def bates_cf(u: np.ndarray, S: float, v0: float, tau: float, r: float, q: float,
             params: BatesParams) -> np.ndarray:
    iu = 1j * u
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho

    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + u ** 2))
    g2 = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    exp_dt = np.exp(-d * tau)
    C_h = (r - q) * iu * tau + kappa * theta / xi ** 2 * (
        (kappa - rho * xi * iu - d) * tau - 2.0 * np.log((1 - g2 * exp_dt) / (1 - g2 + 1e-300)))
    D_h = (kappa - rho * xi * iu - d) / xi ** 2 * (1 - exp_dt) / (1 - g2 * exp_dt + 1e-300)
    cf_heston = np.exp(C_h + D_h * v0 + iu * np.log(S))

    M_u = np.exp(iu * params.mu_J - 0.5 * params.delta_J ** 2 * u ** 2)
    jump_term = np.exp(params.lam * tau * (M_u - 1.0 - params.jump_compensator * iu))

    return cf_heston * jump_term

def cos_call_price_bates(S, K, tau, r, q, v0, params: BatesParams,
                          N_cos: int = 256, L: float = 14.0):
    S = np.atleast_1d(np.asarray(S, dtype=np.float64))
    v0 = np.atleast_1d(np.asarray(v0, dtype=np.float64))
    K = np.atleast_1d(np.asarray(K, dtype=np.float64))
    tau = float(tau)
    P = S.shape[0]
    Nk = K.shape[0]

    sigma_eff = np.sqrt(np.maximum(v0.mean() + params.lam * (params.mu_J ** 2 + params.delta_J ** 2),
                                     1e-8))
    width = L * sigma_eff * np.sqrt(tau) + 5 * sigma_eff * np.sqrt(tau)

    log_S_over_K = np.log(S[:, None] / K[None, :])
    a = float(log_S_over_K.min() - width)
    b = float(log_S_over_K.max() + width)

    k_idx = np.arange(N_cos)
    u = k_idx * np.pi / (b - a)

    cf_vals = np.empty((P, N_cos), dtype=np.complex128)
    for j in range(P):
        cf_vals[j] = bates_cf(u, S[j], v0[j], tau, r, q, params)

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

def bates_call_price_per_day(S, V, K, tau, otype, r, q, params):
    call_p = cos_call_price_bates(S, K, tau, r, q, V, params)
    is_call = (otype == "C")
    put_p = call_p - S[:, None] * np.exp(-q * tau) + K[None, :] * np.exp(-r * tau)
    return np.where(is_call[None, :], call_p, put_p)

def bates_delta_per_day_FD(S, V, K, tau, otype, r, q, params, h_rel=0.01):
    h = h_rel * S
    P_plus = cos_call_price_bates(S + h, K, tau, r, q, V, params)
    P_minus = cos_call_price_bates(S - h, K, tau, r, q, V, params)
    delta_call = (P_plus - P_minus) / (2.0 * h[:, None])
    is_call = (otype == "C")
    delta_put = delta_call - np.exp(-q * tau)
    return np.where(is_call[None, :], delta_call, delta_put)
