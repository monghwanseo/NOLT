import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, r, q, sigma, tau):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-300))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2

def bsm_call_delta(S, K, r, q, sigma, tau):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    return np.exp(-q * tau) * norm.cdf(d1)

def bsm_put_delta(S, K, r, q, sigma, tau):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    return np.exp(-q * tau) * (norm.cdf(d1) - 1.0)

def bsm_delta(S, K, r, q, sigma, tau, option_type):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    is_call = (np.asarray(option_type) == 'C')
    return np.where(is_call,
                    np.exp(-q * tau) * norm.cdf(d1),
                    np.exp(-q * tau) * (norm.cdf(d1) - 1.0))

def bsm_call_equivalent_delta(delta, q, tau, option_type):
    delta = np.asarray(delta, dtype=float)
    is_call = (np.asarray(option_type) == 'C')
    return np.where(is_call, delta, delta + np.exp(-q * tau))

def bsm_gamma(S, K, r, q, sigma, tau):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-300))
    return np.exp(-q * tau) * norm.pdf(d1) / (S * sigma * sqrt_tau)

def bsm_vega(S, K, r, q, sigma, tau):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-300))
    return S * np.exp(-q * tau) * norm.pdf(d1) * sqrt_tau

def bsm_theta(S, K, r, q, sigma, tau, option_type):
    d1, d2 = _d1_d2(S, K, r, q, sigma, tau)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-300))
    term1 = -np.exp(-q * tau) * S * norm.pdf(d1) * sigma / (2 * sqrt_tau)
    is_call = (np.asarray(option_type) == 'C')
    term2 = q * np.exp(-q * tau) * S * np.where(is_call, norm.cdf(d1), -norm.cdf(-d1))
    term3 = -r * K * np.exp(-r * tau) * np.where(is_call, norm.cdf(d2), -norm.cdf(-d2))
    return term1 + term2 + term3
