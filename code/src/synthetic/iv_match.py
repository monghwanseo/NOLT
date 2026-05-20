from __future__ import annotations
import numpy as np
from scipy.stats import norm

from src.synthetic.heston import cos_call_price

def bsm_price_call(S, K, tau, r, q, sigma):
    sqrt_t = np.sqrt(np.maximum(tau, 1e-300))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bsm_price_put(S, K, tau, r, q, sigma):
    sqrt_t = np.sqrt(np.maximum(tau, 1e-300))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)

def bsm_vega(S, K, tau, r, q, sigma):
    sqrt_t = np.sqrt(np.maximum(tau, 1e-300))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_t)
    return S * np.exp(-q * tau) * norm.pdf(d1) * sqrt_t

def implied_vol_newton(target_price, S, K, tau, r, q, otype, sigma_init=0.2,
                       n_iter=40, tol=1e-8):
    sigma = np.full_like(np.asarray(target_price, dtype=np.float64), sigma_init)
    is_call = (otype == "C")
    for _ in range(n_iter):
        call_target = np.where(is_call, target_price,
                                target_price + S * np.exp(-q * tau) - K * np.exp(-r * tau))
        bs_call = bsm_price_call(S, K, tau, r, q, sigma)
        diff = bs_call - call_target
        vega = bsm_vega(S, K, tau, r, q, sigma)
        step = diff / np.maximum(vega, 1e-12)
        step = np.clip(step, -0.5, 0.5)
        sigma = sigma - step
        sigma = np.clip(sigma, 1e-3, 5.0)
        if np.nanmax(np.abs(diff)) < tol:
            break
    return sigma

def heston_price_per_day(S, V, K, tau, otype, r, q, params):
    call_p = cos_call_price(S, K, tau, r, q, V, params)
    is_call = (otype == "C")
    put_p = call_p - S[:, None] * np.exp(-q * tau) + K[None, :] * np.exp(-r * tau)
    return np.where(is_call[None, :], call_p, put_p)

def heston_delta_per_day_FD(S, V, K, tau, otype, r, q, params, h_rel=0.01):
    h = h_rel * S
    P_plus_call = cos_call_price(S + h, K, tau, r, q, V, params)
    P_minus_call = cos_call_price(S - h, K, tau, r, q, V, params)
    delta_call = (P_plus_call - P_minus_call) / (2.0 * h[:, None])
    is_call = (otype == "C")
    delta_put = delta_call - np.exp(-q * tau)
    return np.where(is_call[None, :], delta_call, delta_put)
