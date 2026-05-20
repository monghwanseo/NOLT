import pandas as pd

from .config import QUALITY_THRESHOLDS

GREEK_COLS = ['Implied Volatility Mid', 'Delta Mid Price', 'Gamma Mid']

COVERAGE_COLS = {
    'iv_coverage': 'Implied Volatility Mid',
    'delta_coverage': 'Delta Mid Price',
    'gamma_coverage': 'Gamma Mid',
    'mid_coverage': 'Mid Price',
    'volume_coverage': 'Volume',
    'oi_coverage': 'Open Interest',
}

def build_meta(panel: pd.DataFrame) -> pd.DataFrame:
    grp = panel.groupby('ticker', sort=False)
    meta = grp.agg(
        n_rows=('Date', 'size'),
        date_min=('Date', 'min'),
        date_max=('Date', 'max'),
    ).reset_index()
    first = grp[['option_type', 'strike', 'expiry', 'sheet_name']].first().reset_index()
    meta = meta.merge(first, on='ticker')
    meta['date_span_days'] = (meta['date_max'] - meta['date_min']).dt.days
    meta = meta.sort_values(['expiry', 'option_type', 'strike']).reset_index(drop=True)
    return meta

def compute_coverages(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in panel.groupby('ticker', sort=False):
        n = len(g)
        cov = {'ticker': ticker, 'n_rows': n}
        for cov_name, col in COVERAGE_COLS.items():
            if col in g.columns and n > 0:
                cov[cov_name] = float(g[col].notna().sum()) / n
            else:
                cov[cov_name] = 0.0
        rows.append(cov)
    return pd.DataFrame(rows)

def classify(row: pd.Series) -> str:
    iv = row['iv_coverage']
    d = row['delta_coverage']
    g = row['gamma_coverage']
    mid = row['mid_coverage']
    t = QUALITY_THRESHOLDS
    greeks_ok = (iv >= t['iv_min']) and (d >= t['delta_min']) and (g >= t['gamma_min'])
    if not greeks_ok:
        return 'EXCLUDE'
    if mid >= t['mid_full_threshold']:
        return 'FULL_USE'
    return 'GREEKS_ONLY'

def build_quality_report(panel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cov = compute_coverages(panel)
    cov['classification'] = cov.apply(classify, axis=1)
    qr = cov.merge(
        meta[['ticker', 'option_type', 'strike', 'expiry', 'sheet_name', 'date_min', 'date_max', 'date_span_days']],
        on='ticker',
    )
    qr = qr.sort_values(['expiry', 'option_type', 'strike']).reset_index(drop=True)
    return qr
