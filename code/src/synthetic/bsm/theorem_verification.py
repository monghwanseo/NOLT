import numpy as np
from scipy.stats import norm

from src.metrics.bsm_greeks import (bsm_delta, bsm_gamma, bsm_vega,
                                    bsm_call_equivalent_delta)
from src.metrics.linearity import L_B

def _d1(S, K, r, q, sigma, tau):
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-300))
    return (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)

def L_B_call(S, K, r, q, sigma, tau):
    delta_c = bsm_delta(S, K, r, q, sigma, tau, 'C')
    eq = bsm_call_equivalent_delta(delta_c, q, tau, np.array(['C']*np.broadcast_shapes(*np.broadcast_arrays(S, K, sigma, tau)[:1])[0] if hasattr(S, '__len__') else 1))
    return L_B(eq)

def L_B_call_simple(S, K, r, q, sigma, tau):
    d1 = _d1(S, K, r, q, sigma, tau)
    delta = np.exp(-q * tau) * norm.cdf(d1)
    return (2.0 * delta - 1.0) ** 2

def L_D_typical(S, K, r, q, sigma, tau, dt_typical=1/252):
    delta_call = bsm_delta(S, K, r, q, sigma, tau, 'C')
    G = bsm_gamma(S, K, r, q, sigma, tau)
    dS_typ = sigma * S * np.sqrt(dt_typical)
    convex = 0.5 * np.abs(G) * dS_typ
    denom = np.abs(delta_call)
    LD = 1.0 - convex / np.maximum(denom, 1e-12)
    return np.clip(LD, 0.0, 1.0)

def regress_T2(K_grid, tau_grid, S=100.0, r=0.04, q=0.0117, sigma=0.20):
    import statsmodels.api as sm
    K, TAU = np.meshgrid(K_grid, tau_grid)
    K = K.flatten(); TAU = TAU.flatten()
    LBval = L_B_call_simple(S, K, r, q, sigma, TAU)
    one_minus_LB = 1.0 - LBval

    mask = (one_minus_LB > 1e-10) & (one_minus_LB < 0.99) & (TAU > 1e-6)
    inv_tau = 1.0 / TAU[mask]
    log_one_minus = np.log(one_minus_LB[mask])
    Xx = sm.add_constant(inv_tau)
    fit = sm.OLS(log_one_minus, Xx).fit()
    return {
        'inv_tau': inv_tau,
        'log_1_minus_LB': log_one_minus,
        'K': K[mask],
        'tau': TAU[mask],
        'slope': float(fit.params[1]),
        'intercept': float(fit.params[0]),
        'r2': float(fit.rsquared),
    }

def L_B_along_path(S_path, K, r, q, sigma, T, times):
    n_paths, n_steps_plus_1 = S_path.shape
    LB = np.zeros_like(S_path)
    for k in range(n_steps_plus_1):
        tau = max(T - times[k], 1e-8)
        LB[:, k] = L_B_call_simple(S_path[:, k], K, r, q, sigma, tau)
    return LB
