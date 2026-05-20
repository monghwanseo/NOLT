import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(2026)

from src.data import config as cfg
from src.data.tasks import (
    common_dates, window_a_tickers, window_b_tickers, window_c_tickers,
)

PROC = cfg.PROCESSED_DIR
FIG = cfg.FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)

panel = pd.read_parquet(PROC / 'options_panel.parquet')
meta = pd.read_parquet(PROC / 'options_meta.parquet')
qr = pd.read_csv(PROC / 'quality_report.csv', parse_dates=['expiry', 'date_min', 'date_max'])
spx_pcp = pd.read_parquet(PROC / 'spx_pcp.parquet')
q_imp = pd.read_parquet(PROC / 'q_implied.parquet')
pcp = pd.read_parquet(PROC / 'pcp_pairs.parquet')

ta = window_a_tickers(qr)
tb = window_b_tickers(qr)
tc = window_c_tickers(meta, qr)
cdA = sorted(common_dates(panel, ta))
cdB = sorted(common_dates(panel, tb))
cdC = sorted(common_dates(panel, tc))

print("=== Generating figures ===")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(spx_pcp['Date'], spx_pcp['S_pcp'], lw=0.8, color='#1f4e79')
ax.fill_between(spx_pcp['Date'],
                spx_pcp['S_pcp'] - spx_pcp['S_pcp_std'].fillna(0),
                spx_pcp['S_pcp'] + spx_pcp['S_pcp_std'].fillna(0),
                color='#1f4e79', alpha=0.15, label='+/- cross-pair std')
ax.set_xlabel('Date')
ax.set_ylabel('PCP-implied SPX')
ax.set_title(f'PCP-implied SPX (q=0 bootstrap, r={cfg.R}) -{len(spx_pcp)} dates, mean={spx_pcp["S_pcp"].mean():.0f}')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(FIG / 'pcp_spx.png', dpi=140)
plt.close()
print(f"  pcp_spx.png")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(q_imp['Date'], q_imp['q_implied'], lw=0.8, color='#aa3333', label='per-date median q')
ax.axhline(cfg.Q_BASELINE, color='black', ls='--', lw=1.0, label=f'M10 baseline q={cfg.Q_BASELINE}')
ax.axhline(q_imp['q_implied'].median(), color='#1f4e79', ls=':', lw=1.0,
           label=f'observed median q={q_imp["q_implied"].median():.4f}')
ax.set_xlabel('Date')
ax.set_ylabel('q_implied')
ax.set_title(f'PCP cross-maturity implied q -{len(q_imp)} dates, median={q_imp["q_implied"].median():.4f}')
ax.legend()
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(FIG / 'q_implied.png', dpi=140)
plt.close()
print(f"  q_implied.png")

cov_cols = ['iv_coverage', 'delta_coverage', 'gamma_coverage',
            'mid_coverage', 'volume_coverage', 'oi_coverage']
qr_sorted = qr.sort_values(['expiry', 'option_type', 'strike']).reset_index(drop=True)
heat = qr_sorted[cov_cols].values

fig, ax = plt.subplots(figsize=(8, 14))
im = ax.imshow(heat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

labels = [f"{r['expiry'].strftime('%y-%m')} {r['option_type']}{r['strike']:>5}  [{r['classification'][:3]}]"
          for _, r in qr_sorted.iterrows()]
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xticks(range(len(cov_cols)))
ax.set_xticklabels([c.replace('_coverage', '') for c in cov_cols], rotation=20, ha='right')

plt.colorbar(im, ax=ax, label='coverage (non-NaN ratio)', shrink=0.6)
ax.set_title('Per-option data coverage (sorted by expiry, type, strike)\nGreen >= 0.95, Yellow ~ 0.5, Red ~ 0')
plt.tight_layout()
plt.savefig(FIG / 'quality_heatmap.png', dpi=140)
plt.close()
print(f"  quality_heatmap.png")

fig, ax = plt.subplots(figsize=(11, 14))
classification_color = {'FULL_USE': '#2c8a3a', 'GREEKS_ONLY': '#d4a017', 'EXCLUDE': '#aa3333'}
for i, (_, row) in enumerate(qr_sorted.iterrows()):
    color = classification_color[row['classification']]
    ax.barh(i, (row['date_max'] - row['date_min']).days,
            left=row['date_min'], color=color, alpha=0.85, height=0.7)

ax.set_yticks(range(len(qr_sorted)))
ax.set_yticklabels([f"{r['expiry'].strftime('%y-%m')} {r['option_type']}{r['strike']}"
                    for _, r in qr_sorted.iterrows()], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Date')
ax.set_title('Per-option active date range by classification')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
handles = [Patch(facecolor=c, label=k) for k, c in classification_color.items()]
ax.legend(handles=handles, loc='lower right')
plt.tight_layout()
plt.savefig(FIG / 'expiry_timeline.png', dpi=140)
plt.close()
print(f"  expiry_timeline.png")

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, cv in zip(axes.flat, cov_cols):
    ax.hist(qr[cv], bins=20, color='#1f4e79', edgecolor='black')
    ax.axvline(0.95, color='red', ls='--', lw=1, label='0.95 thresh')
    ax.set_title(cv.replace('_coverage', ''))
    ax.set_xlabel('coverage')
    ax.set_xlim(0, 1.02)
    ax.grid(alpha=0.3)
    if ax is axes.flat[0]:
        ax.legend()
plt.tight_layout()
plt.savefig(FIG / 'coverage_distribution.png', dpi=140)
plt.close()
print(f"  coverage_distribution.png")

print("\n=== Generating data_quality_report.md ===")

window_stats = []
for win, tickers, common in [('A', ta, cdA), ('B', tb, cdB), ('C', tc, cdC)]:
    rows = panel[panel['ticker'].isin(tickers)]
    window_stats.append({
        'window': win,
        'n_options': len(tickers),
        'common_days': len(common),
        'common_start': min(common).isoformat() if common else 'n/a',
        'common_end': max(common).isoformat() if common else 'n/a',
        'panel_rows': len(rows),
    })

excl = qr[qr['classification'] == 'EXCLUDE'][['ticker', 'iv_coverage', 'delta_coverage',
                                               'gamma_coverage', 'mid_coverage', 'n_rows']]

class_by_exp = qr.groupby([qr['expiry'].dt.strftime('%Y-%m-%d'),
                           'classification']).size().unstack(fill_value=0)

md = f"""# NOLT -Data Quality Report (v1)

Generated from `Bloomberg/spx_2.xlsx` via `scripts/01_build_data.py`.

## Summary

| Item | Value |
|---|---|
| Total option sheets | {len(meta)} |
| FULL_USE | {(qr['classification'] == 'FULL_USE').sum()} |
| GREEKS_ONLY | {(qr['classification'] == 'GREEKS_ONLY').sum()} |
| EXCLUDE | {(qr['classification'] == 'EXCLUDE').sum()} |
| Total panel rows | {len(panel):,} |
| Date range (any option) | {panel['Date'].min().date()} to {panel['Date'].max().date()} |
| PCP-implied SPX dates | {len(spx_pcp)} |
| q_implied dates | {len(q_imp)} |
| q_implied median (cross-maturity) | **{q_imp['q_implied'].median():.4f}** (M10 baseline 0.0117) |
| PCP daily change > 10% rate | {((spx_pcp.sort_values("Date")["S_pcp"].pct_change().abs() > 0.10).mean()):.4f} |

## Classification breakdown by expiry

```
{class_by_exp.to_string()}
```

## Excluded options

{excl.to_markdown(index=False) if len(excl) else "_None._"}

## Task windows

| Window | Options | Common days | Date range (intersection) | Use |
|---|---|---|---|---|
"""
for ws in window_stats:
    use = {'A': 'Primary (M5-M10 reproduction)',
           'B': 'Extended cross-section (LEAPS effect)',
           'C': 'Long-history (latent state, ACF)'}[ws['window']]
    md += f"| {ws['window']} | {ws['n_options']} | {ws['common_days']} | {ws['common_start']} -> {ws['common_end']} | {use} |\n"

md += f"""

## Standards (LOCKED)

- Risk-free rate: **r = {cfg.R}**
- Dividend yield: **q = q_implied (per-date) | 0.0117 (M10 fallback)** -q=0 forbidden
- Random seed (all stochastic ops): **{cfg.SEED}**
- Train/Val/Test: **{cfg.TRAIN_VAL_TEST[0]:.0%}/{cfg.TRAIN_VAL_TEST[1]:.0%}/{cfg.TRAIN_VAL_TEST[2]:.0%}**

## Underlying construction (PCP-implied)

Bloomberg's SPX index column is empty for the option sheets, so SPX is reconstructed
from put-call parity:

  S_implied(tau, q=0) = (C - P) + K * exp(-r * tau)
  q_implied (cross-maturity) = -log(S_implied(tau1) / S_implied(tau2)) / (tau1 - tau2)

Cross-maturity pairs require |tau1 - tau2| >= {cfg.Q_MIN_DTAU} years for stability.

## Task panel sizes

| Panel | Rows | Options |
|---|---|---|
| `options_panel.parquet` (master) | {len(panel):,} | {meta['ticker'].nunique()} |
| `greeks_panel_A.parquet` | (see Window A) | {len(ta)} |
| `greeks_panel_B.parquet` | (see Window B) | {len(tb)} |
| `greeks_panel_C.parquet` | (see Window C) | {len(tc)} |
| `pcp_pairs.parquet` | {len(pcp):,} | (Date,K,exp) |
| `hedging_panel.parquet` | {len(panel[panel['ticker'].isin(qr[qr['classification']=='FULL_USE']['ticker'])]):,} | {(qr['classification']=='FULL_USE').sum()} (FULL_USE) |
| `spx_pcp.parquet` | {len(spx_pcp)} | underlying |
| `q_implied.parquet` | {len(q_imp)} | dividend yield |

## Figures

- `figures/pcp_spx.png` -PCP-implied SPX time series with cross-pair std band
- `figures/q_implied.png` -Per-date implied dividend yield, with M10 baseline reference
- `figures/quality_heatmap.png` -Per-option coverage of IV/Delta/Gamma/Mid/Volume/OI
- `figures/expiry_timeline.png` -Active date range per option, colored by classification
- `figures/coverage_distribution.png` -Histogram of each coverage metric

## Limitations (paper §Limitations draft)

- The 56 options were curated by the data provider, **not a systematic sample** - strike grid (5500/6100/6500/6800/7100/7500/8200) is sparse and uniform across
  4 expiries (06/18/26, 12/18/26, 12/17/27, 12/15/28).
- One option excluded for low Greeks coverage (12/18/26 P8200, deep OTM put).
- Daily granularity (business days only); weekend forward-fill not applied to
  preserve unbiased return statistics.
- Underlying SPX inferred from PCP rather than observed directly (Bloomberg
  did not deliver the index time series for these sheets).
- **Future work**: validate on OptionMetrics or comprehensive option databases.
"""

(cfg.DATA_DIR / 'data_quality_report.md').write_text(md, encoding='utf-8')
print(f" -> {cfg.DATA_DIR / 'data_quality_report.md'}")

print("\nDone.")
