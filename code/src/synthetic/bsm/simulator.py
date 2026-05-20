import numpy as np

def gbm_paths(S0: float, r: float, q: float, sigma: float,
              T: float, n_steps: int, n_paths: int,
              rng: np.random.Generator | int = 2026) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    mu = r - q
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diff = sigma * np.sqrt(dt) * Z
    log_increments = drift + diff
    log_S_path = np.log(S0) + np.cumsum(log_increments, axis=1)
    S0_col = np.full((n_paths, 1), S0)
    S = np.concatenate([S0_col, np.exp(log_S_path)], axis=1)
    times = np.linspace(0.0, T, n_steps + 1)
    return S, times

def annual_to_daily_steps(T_years: float, days_per_year: int = 252) -> int:
    return max(1, int(round(T_years * days_per_year)))
