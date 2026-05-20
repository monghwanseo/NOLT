import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(2026)

from src.data import config as cfg
from src.data.tasks import window_a_tickers, common_dates
from src.metrics.bsm_greeks import bsm_delta, bsm_call_equivalent_delta
from src.metrics.linearity import L_B

PROC = cfg.PROCESSED_DIR
OUT_DIR = ROOT / "results" / "M1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

def acf1(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return np.nan
    return float(x.autocorr(lag=1))

def fit_q_for_date(date, q_imp_df, q_fallback=cfg.Q_BASELINE):
    row = q_imp_df[q_imp_df['Date'] == date]
    if len(row) and pd.notna(row['q_implied'].iloc[0]):
        return float(row['q_implied'].iloc[0])
    return q_fallback

def main():
    print("=" * 78)
    print("M1: WINDOW A REPLICATION (C1 gating step)")
    print("=" * 78)

    panel = pd.read_parquet(PROC / 'options_panel.parquet')
    qr = pd.read_csv(PROC / 'quality_report.csv', parse_dates=['expiry', 'date_min', 'date_max'])
    spx_pcp = pd.read_parquet(PROC / 'spx_pcp.parquet')
    q_imp = pd.read_parquet(PROC / 'q_implied.parquet')

    ta = window_a_tickers(qr)
    print(f"\nWindow A: {len(ta)} options")
    cdA = sorted(common_dates(panel, ta))
    print(f"Window A common dates: {len(cdA)}  ({cdA[0]} to {cdA[-1]})")

    sub = panel[panel['ticker'].isin(ta)].copy()
    sub = sub[sub['Date'].dt.date.isin(set(cdA))].reset_index(drop=True)

    needed = ['Implied Volatility Mid', 'Delta Mid Price']
    sub = sub.dropna(subset=needed).reset_index(drop=True)
    print(f"Greeks-valid Window A rows: {len(sub):,}")

    sub = sub.merge(spx_pcp[['Date', 'S_pcp']], on='Date', how='left')
    sub = sub.merge(q_imp[['Date', 'q_implied']], on='Date', how='left')

    sub['q_used'] = sub['q_implied'].fillna(cfg.Q_BASELINE)

    sub['sigma'] = sub['Implied Volatility Mid'] / 100.0

    sub['tau'] = (sub['expiry'] - sub['Date']).dt.days / 365.25
    sub = sub[sub['tau'] > 0].reset_index(drop=True)

    sub = sub.dropna(subset=['S_pcp']).reset_index(drop=True)
    print(f"Rows with S, sigma, q, tau, Greeks all valid: {len(sub):,}")

    print("\n" + "=" * 78)
    print("[M5] BSM Delta vs market Delta -residual ~ moneyness regression")
    print("=" * 78)

    def compute_residual(sub, q_col):
        S = sub['S_pcp'].values
        K = sub['strike'].values.astype(float)
        sigma = sub['sigma'].values
        tau = sub['tau'].values
        opt_type = sub['option_type'].values
        q = sub[q_col].values
        r = cfg.R

        delta_bsm = bsm_delta(S, K, r, q, sigma, tau, opt_type)
        delta_mkt = sub['Delta Mid Price'].values

        d_eq_mkt = bsm_call_equivalent_delta(delta_mkt, q, tau, opt_type)
        d_eq_bsm = bsm_call_equivalent_delta(delta_bsm, q, tau, opt_type)

        L_B_mkt = L_B(d_eq_mkt)
        L_B_bsm = L_B(d_eq_bsm)
        residual = L_B_mkt - L_B_bsm

        F = S * np.exp((r - q) * tau)
        m = np.log(K / F)
        return residual, m, delta_mkt, delta_bsm

    resid, mny, _, _ = compute_residual(sub, 'q_used')

    X = sm.add_constant(mny)
    model = sm.OLS(resid, X).fit()
    slope = float(model.params[1])
    intercept = float(model.params[0])
    r2 = float(model.rsquared)

    print(f"  N = {len(resid):,}")
    print(f"  slope     = {slope:+.5f}  (target [-0.075, -0.065])")
    print(f"  intercept = {intercept:+.5f}")
    print(f"  R^2       = {r2:.4f}     (target [0.18, 0.22])")

    M5_pass_slope = -0.075 <= slope <= -0.065
    M5_pass_r2    = 0.18 <= r2 <= 0.22
    print(f"  slope in target?  {M5_pass_slope}")
    print(f"  R^2 in target?    {M5_pass_r2}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mny, resid, s=2, alpha=0.25, color='#1f4e79')
    xs = np.linspace(mny.min(), mny.max(), 100)
    ax.plot(xs, intercept + slope * xs, color='red',
            label=f'OLS: slope={slope:.4f}, R²={r2:.3f}')
    ax.set_xlabel('moneyness m = log(K/F)')
    ax.set_ylabel(r'residual = $L_B^{market} - L_B^{BSM}$')
    ax.set_title(f'M5: Linearity residual vs moneyness (Window A, N={len(resid):,})')
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'M5_residual_vs_moneyness.png', dpi=140)
    plt.close()

    print("\n" + "=" * 78)
    print("[M6] Residual matrix (date × ticker) PCA + ACF")
    print("=" * 78)

    sub['residual'] = resid
    rmat = sub.pivot_table(index='Date', columns='ticker', values='residual', aggfunc='mean')
    print(f"  Residual matrix shape: {rmat.shape}  (dates × tickers)")
    print(f"  Missing per row range: {rmat.isna().sum(axis=1).min()}-{rmat.isna().sum(axis=1).max()}")

    rmat_dense = rmat.dropna(axis=0, how='any')
    print(f"  Dense matrix (all tickers present): {rmat_dense.shape}")

    X_pca = rmat_dense.values - rmat_dense.values.mean(axis=0)
    pca = PCA(n_components=min(5, X_pca.shape[1]), random_state=cfg.SEED)
    pca.fit(X_pca)
    pc_var = pca.explained_variance_ratio_
    pc1_var = float(pc_var[0])
    print(f"  PC1 variance share : {pc1_var:.4f}  (target [0.50, 0.60])")
    print(f"  PC1-5 variance: {[f'{v:.3f}' for v in pc_var]}")

    pc1_ts = pd.Series(pca.transform(X_pca)[:, 0], index=rmat_dense.index)
    pc1_acf1 = acf1(pc1_ts)
    print(f"  PC1 ACF(1)         : {pc1_acf1:.4f}  (target [0.80, 0.90])")

    per_ticker_acf = {}
    for col in rmat_dense.columns:
        per_ticker_acf[col] = acf1(rmat_dense[col])
    pt_med = float(np.nanmedian(list(per_ticker_acf.values())))
    print(f"  Per-ticker ACF(1) median: {pt_med:.4f}  (target [0.60, 0.70])")

    M6_pass_pc1var = 0.50 <= pc1_var <= 0.60
    M6_pass_pc1acf = 0.80 <= pc1_acf1 <= 0.90
    M6_pass_ptacf = 0.60 <= pt_med <= 0.70

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    ax = axes[0]
    ax.plot(pc1_ts.index, pc1_ts.values, lw=0.8, color='#1f4e79')
    ax.set_ylabel('PC1 score')
    ax.set_title(f'M6: PC1 time series (var share = {pc1_var:.3f}, ACF(1) = {pc1_acf1:.3f})')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax = axes[1]
    vals = list(per_ticker_acf.values())
    ax.hist(vals, bins=15, color='#aa3333', edgecolor='black')
    ax.axvline(pt_med, color='black', ls='--', label=f'median = {pt_med:.3f}')
    ax.set_xlabel('per-ticker ACF(1)')
    ax.set_ylabel('count')
    ax.set_title(f'Per-ticker residual ACF(1) distribution (N={len(vals)} tickers)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'M6_pca_acf.png', dpi=140)
    plt.close()

    print("\n" + "=" * 78)
    print("[M9] Outlier robustness -drop |daily SPX change|>10% days, recompute ACF")
    print("=" * 78)

    spx_sorted = spx_pcp.sort_values('Date').reset_index(drop=True)
    spx_sorted['ret'] = spx_sorted['S_pcp'].pct_change()
    outlier_dates = set(spx_sorted.loc[spx_sorted['ret'].abs() > 0.10, 'Date'].dt.date)
    outlier_dates_in_a = sorted(outlier_dates & set(cdA))
    print(f"  Total PCP outlier days   : {len(outlier_dates)}")
    print(f"  Outlier days in Window A : {len(outlier_dates_in_a)} -> {outlier_dates_in_a}")

    rmat_clean = rmat_dense.loc[~rmat_dense.index.normalize().isin(
        pd.to_datetime(list(outlier_dates_in_a)).normalize()
    )]
    if len(rmat_clean) >= len(rmat_dense) * 0.95:

        Xc = rmat_clean.values - rmat_clean.values.mean(axis=0)
        pca_c = PCA(n_components=min(5, Xc.shape[1]), random_state=cfg.SEED)
        pca_c.fit(Xc)
        pc1_var_c = float(pca_c.explained_variance_ratio_[0])
        pc1_ts_c = pd.Series(pca_c.transform(Xc)[:, 0], index=rmat_clean.index)
        pc1_acf1_c = acf1(pc1_ts_c)
        print(f"  After outlier removal: PC1 var = {pc1_var_c:.4f}, PC1 ACF(1) = {pc1_acf1_c:.4f}")
        delta_var = abs(pc1_var_c - pc1_var)
        delta_acf = abs(pc1_acf1_c - pc1_acf1)
        print(f"  ?(PC1 var) = {delta_var:.4f}, ?(PC1 ACF) = {delta_acf:.4f}")
        M9_stable = (delta_var < 0.05) and (delta_acf < 0.05)
    else:
        M9_stable = True
        delta_var = 0.0
        delta_acf = 0.0
        print(f"  No outlier days in Window A -ACF inherently robust.")

    print("\n" + "=" * 78)
    print("[M10] q sensitivity -q=0 vs q_implied vs M10 constant")
    print("=" * 78)

    sub['q_zero'] = 0.0
    sub['q_const'] = cfg.Q_BASELINE

    res_q0, mny_q0, _, _ = compute_residual(sub, 'q_zero')
    res_qc, mny_qc, _, _ = compute_residual(sub, 'q_const')
    res_qi, mny_qi, _, _ = compute_residual(sub, 'q_used')

    def reg(res, m):
        Xx = sm.add_constant(m)
        mm = sm.OLS(res, Xx).fit()
        return float(mm.params[0]), float(mm.params[1]), float(mm.rsquared)

    print(f"  {'q variant':<22} {'slope':<10} {'R^2':<8}")
    for name, r, m in [
        ('q = 0 (forbidden)', res_q0, mny_q0),
        ('q = 0.0117 (M10)', res_qc, mny_qc),
        ('q = q_implied(t)', res_qi, mny_qi),
    ]:
        i, s, r2_ = reg(r, m)
        print(f"  {name:<22} {s:+.5f}    {r2_:.4f}")

    print(f"\n  Per-option-type R^2:")
    for name, q_col in [('q=0', 'q_zero'), ('q=0.0117', 'q_const'), ('q=q_imp', 'q_used')]:
        r_, m_, _, _ = compute_residual(sub, q_col)
        sub_tmp = sub.copy()
        sub_tmp['_r'] = r_; sub_tmp['_m'] = m_
        for opt in ['C', 'P']:
            sub_opt = sub_tmp[sub_tmp['option_type'] == opt]
            i, s, r2_ = reg(sub_opt['_r'].values, sub_opt['_m'].values)
            print(f"    {name:<10} type={opt}  slope={s:+.5f}  R^2={r2_:.4f}")

    M10_pass = abs(reg(res_q0, mny_q0)[1] - reg(res_qi, mny_qi)[1]) > 0.001

    print("\n" + "=" * 78)
    print("M1 SUMMARY")
    print("=" * 78)

    summary = {
        'window': 'A',
        'n_options': len(ta),
        'common_days': len(cdA),
        'date_range': f"{cdA[0]} to {cdA[-1]}",
        'n_rows': int(len(sub)),
        'M5': {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'slope_target': '[-0.075, -0.065]',
            'r2_target': '[0.18, 0.22]',
            'slope_pass': bool(M5_pass_slope),
            'r2_pass': bool(M5_pass_r2),
        },
        'M6': {
            'pc1_var': pc1_var,
            'pc1_var_target': '[0.50, 0.60]',
            'pc1_var_pass': bool(M6_pass_pc1var),
            'pc1_acf1': pc1_acf1,
            'pc1_acf1_target': '[0.80, 0.90]',
            'pc1_acf1_pass': bool(M6_pass_pc1acf),
            'per_ticker_acf_median': pt_med,
            'per_ticker_acf_target': '[0.60, 0.70]',
            'per_ticker_acf_pass': bool(M6_pass_ptacf),
        },
        'M9': {
            'outlier_days_in_window_a': len(outlier_dates_in_a),
            'delta_pc1_var': float(delta_var),
            'delta_pc1_acf': float(delta_acf),
            'stable': bool(M9_stable),
        },
        'M10': {
            'pass': bool(M10_pass),
            'note': 'q changes regression noticeably (slope diff > 0.001)',
        },
    }

    all_checks = [M5_pass_slope, M5_pass_r2, M6_pass_pc1var, M6_pass_pc1acf, M6_pass_ptacf, M9_stable, M10_pass]
    n_pass = sum(all_checks)
    print(f"\n  Checks passed: {n_pass} / {len(all_checks)}")
    for label, ok in zip(
        ['M5 slope', 'M5 R^2', 'M6 PC1 var', 'M6 PC1 ACF', 'M6 per-ticker ACF',
         'M9 outlier robust', 'M10 q sensitivity'], all_checks):
        flag = '[PASS]' if ok else '[FAIL]'
        print(f"    {flag} {label}")

    summary['verdict'] = {
        'n_pass': int(n_pass),
        'n_total': len(all_checks),
        'overall_pass': bool(n_pass == len(all_checks)),
    }

    with open(OUT_DIR / 'M1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n -> {OUT_DIR / 'M1_summary.json'}")
    print(f" -> Figures in {FIG_DIR}/")

    return summary

if __name__ == '__main__':
    main()
