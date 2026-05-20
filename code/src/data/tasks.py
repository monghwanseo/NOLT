import pandas as pd

from .config import WINDOW_A_EXPIRIES, WINDOW_C_MIN_DAYS

GREEK_COLS = ('Implied Volatility Mid', 'Delta Mid Price', 'Gamma Mid')

def window_a_tickers(quality_report: pd.DataFrame) -> list[str]:
    qr = quality_report
    expiries = pd.to_datetime(WINDOW_A_EXPIRIES, format='%m/%d/%y')
    mask = qr['expiry'].isin(expiries) & qr['classification'].isin(['FULL_USE', 'GREEKS_ONLY'])
    return qr.loc[mask, 'ticker'].tolist()

def window_b_tickers(quality_report: pd.DataFrame) -> list[str]:
    return quality_report.loc[
        quality_report['classification'].isin(['FULL_USE', 'GREEKS_ONLY']),
        'ticker',
    ].tolist()

def window_c_tickers(meta: pd.DataFrame, quality_report: pd.DataFrame, min_days: int = WINDOW_C_MIN_DAYS) -> list[str]:
    qr = quality_report
    valid = set(qr.loc[qr['classification'] != 'EXCLUDE', 'ticker'])
    long_hist = set(meta.loc[meta['date_span_days'] >= min_days, 'ticker'])
    selected = sorted(valid & long_hist)
    return selected

def build_greeks_panel(panel: pd.DataFrame, tickers: list[str], greek_cols=GREEK_COLS) -> pd.DataFrame:
    sub = panel[panel['ticker'].isin(tickers)]
    cols = [c for c in greek_cols if c in sub.columns]
    if not cols:
        return sub.iloc[0:0].copy()
    mask = sub[cols].notna().all(axis=1)
    return sub.loc[mask].reset_index(drop=True)

def build_pcp_pairs(panel: pd.DataFrame, full_use_tickers: list[str]) -> pd.DataFrame:
    sub = panel[panel['ticker'].isin(full_use_tickers)].copy()
    if 'Mid Price' not in sub.columns:
        return pd.DataFrame(columns=['Date', 'strike', 'expiry', 'C_mid', 'P_mid'])
    calls = sub[sub['option_type'] == 'C'][['Date', 'strike', 'expiry', 'Mid Price']]
    puts  = sub[sub['option_type'] == 'P'][['Date', 'strike', 'expiry', 'Mid Price']]
    pairs = (
        calls.rename(columns={'Mid Price': 'C_mid'})
        .merge(puts.rename(columns={'Mid Price': 'P_mid'}),
               on=['Date', 'strike', 'expiry'], how='inner')
    )
    pairs = pairs.dropna(subset=['C_mid', 'P_mid']).reset_index(drop=True)
    pairs = pairs.sort_values(['Date', 'expiry', 'strike']).reset_index(drop=True)
    return pairs

def build_hedging_panel(panel: pd.DataFrame, full_use_tickers: list[str]) -> pd.DataFrame:
    sub = panel[panel['ticker'].isin(full_use_tickers)]
    if 'Mid Price' not in sub.columns:
        return sub.iloc[0:0].copy()
    return sub[sub['Mid Price'].notna()].reset_index(drop=True)

def common_dates(panel: pd.DataFrame, tickers: list[str]) -> set:
    if not tickers:
        return set()
    sets = [set(panel.loc[panel['ticker'] == t, 'Date'].dt.date.unique()) for t in tickers]
    return set.intersection(*sets) if sets else set()
