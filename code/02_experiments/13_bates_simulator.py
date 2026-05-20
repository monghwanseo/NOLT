from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.synthetic.bates import (
    BatesParams, simulate_bates, cos_call_price_bates,
    bates_call_price_per_day, bates_delta_per_day_FD,
)

if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    p = BatesParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04,
                     lam=2.0, mu_J=-0.05, delta_J=0.10)
    print(f"Bates params: {p}")
    print(f"Jump compensator: {p.jump_compensator:.6f}")

    S, V = simulate_bates(p, 100.0, 0.04, 0.0117, 30, 5, rng)
    print(f"S shape={S.shape}; S[:, 0]={S[:, 0]}; S[:, -1]={S[:, -1]}")
    print(f"V[:, -1]={V[:, -1]}")

    K_test = np.array([90.0, 100.0, 110.0])
    prices = cos_call_price_bates(S[:, 0], K_test, 2.0, 0.04, 0.0117, V[:, 0], p)
    print(f"Bates call prices (5 paths x 3 strikes, tau=2):\n{prices}")

    p_no_jump = BatesParams(**{**p.__dict__, "lam": 0.0})
    prices_noj = cos_call_price_bates(S[:, 0], K_test, 2.0, 0.04, 0.0117, V[:, 0], p_no_jump)
    print(f"\nBates (lam=0, equivalent to Heston) call prices:\n{prices_noj}")

    print(f"\nJump premium at K=110 (OTM call): {prices[0, 2] - prices_noj[0, 2]:+.4f}")
    print(f"Jump premium at K=90 (ITM call):  {prices[0, 0] - prices_noj[0, 0]:+.4f}")
