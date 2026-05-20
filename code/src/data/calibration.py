import numpy as np
import pandas as pd

from .config import Q_MIN_DTAU, R

def compute_pcp_implied_spot(pcp_pairs: pd.DataFrame, r: float = R) -> pd.DataFrame:
    df = pcp_pairs.copy()
    df['tau'] = (df['expiry'] - df['Date']).dt.days / 365.25
    df = df[df['tau'] > 0].copy()

    df['S_implied'] = df['C_mid'] - df['P_mid'] + df['strike'] * np.exp(-r * df['tau'])

    spx = (
        df.groupby('Date')
        .agg(
            S_pcp=('S_implied', 'mean'),
            S_pcp_std=('S_implied', 'std'),
            n_pairs=('S_implied', 'size'),
        )
        .reset_index()
        .sort_values('Date')
        .reset_index(drop=True)
    )
    return spx

def compute_q_implied(pcp_pairs: pd.DataFrame, r: float = R, min_dtau: float = Q_MIN_DTAU) -> pd.DataFrame:
    df = pcp_pairs.copy()
    df['tau'] = (df['expiry'] - df['Date']).dt.days / 365.25
    df = df[df['tau'] > 0].copy()

    df['S_implied'] = df['C_mid'] - df['P_mid'] + df['strike'] * np.exp(-r * df['tau'])

    by_exp = (
        df.groupby(['Date', 'expiry'])
        .agg(S=('S_implied', 'mean'), tau=('tau', 'first'))
        .reset_index()
    )

    out = []
    for date, g in by_exp.groupby('Date'):
        g = g.sort_values('tau').reset_index(drop=True)
        if len(g) < 2:
            continue
        qs = []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                tau1, tau2 = g.loc[i, 'tau'], g.loc[j, 'tau']
                if abs(tau1 - tau2) < min_dtau:
                    continue
                S1, S2 = g.loc[i, 'S'], g.loc[j, 'S']
                if S1 <= 0 or S2 <= 0:
                    continue

                q = -np.log(S1 / S2) / (tau1 - tau2)
                qs.append(q)
        if qs:
            out.append({
                'Date': date,
                'q_implied': float(np.median(qs)),
                'q_std': float(np.std(qs)) if len(qs) > 1 else 0.0,
                'n_pairs': len(qs),
            })
    out_df = pd.DataFrame(out).sort_values('Date').reset_index(drop=True)
    return out_df
