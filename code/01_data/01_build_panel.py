import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd

np.random.seed(2026)

from src.data import config as cfg
from src.data.calibration import compute_pcp_implied_spot, compute_q_implied
from src.data.loader import load_workbook
from src.data.quality import build_meta, build_quality_report
from src.data.tasks import (
    build_greeks_panel,
    build_hedging_panel,
    build_pcp_pairs,
    common_dates,
    window_a_tickers,
    window_b_tickers,
    window_c_tickers,
)
from src.data.validate import print_sanity_results, sanity_checks

def main():
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[Step 2] Loading {cfg.RAW_XLSX.name} ...")
    print("=" * 78)
    panel = load_workbook(cfg.RAW_XLSX)
    print(f"  panel rows : {len(panel):,}")
    print(f"  columns    : {list(panel.columns)}")
    print(f"  date range : {panel['Date'].min().date()} to {panel['Date'].max().date()}")
    panel.to_parquet(cfg.PROCESSED_DIR / 'options_panel.parquet')
    print(f"  -> options_panel.parquet")

    print(f"\n[Step 3] Building meta ...")
    meta = build_meta(panel)
    meta.to_csv(cfg.PROCESSED_DIR / 'options_meta.csv', index=False)
    meta.to_parquet(cfg.PROCESSED_DIR / 'options_meta.parquet')
    print(f"  -> options_meta.{{csv,parquet}}  ({len(meta)} options)")

    print(f"\n[Step 4] Quality classification ...")
    qr = build_quality_report(panel, meta)
    qr.to_csv(cfg.PROCESSED_DIR / 'quality_report.csv', index=False)
    print(qr['classification'].value_counts().to_string())

    print(f"\n[Step 5] Task panels ...")
    full_use = qr.loc[qr['classification'] == 'FULL_USE', 'ticker'].tolist()
    ta = window_a_tickers(qr)
    tb = window_b_tickers(qr)
    tc = window_c_tickers(meta, qr)
    print(f"  Window A (06/26+12/26, FULL+GO)  : {len(ta)} options")
    print(f"  Window B (all FULL+GO)           : {len(tb)} options")
    print(f"  Window C (long-history)          : {len(tc)} options")
    print(f"  FULL_USE total                   : {len(full_use)}")

    for win, tickers in [('A', ta), ('B', tb), ('C', tc)]:
        gp = build_greeks_panel(panel, tickers)
        out = cfg.PROCESSED_DIR / f'greeks_panel_{win}.parquet'
        gp.to_parquet(out)
        print(f"  greeks_panel_{win} : {len(gp):>7,} rows  ({gp['ticker'].nunique()} options)")

    pcp = build_pcp_pairs(panel, full_use)
    pcp.to_parquet(cfg.PROCESSED_DIR / 'pcp_pairs.parquet')
    print(f"  pcp_pairs       : {len(pcp):>7,} (Date,strike,expiry) rows")

    hp = build_hedging_panel(panel, full_use)
    hp.to_parquet(cfg.PROCESSED_DIR / 'hedging_panel.parquet')
    print(f"  hedging_panel   : {len(hp):>7,} rows")

    cdA = common_dates(panel, ta)
    cdB = common_dates(panel, tb)
    cdC = common_dates(panel, tc)
    print(f"\n  Common dates (intersection):")
    print(f"    Window A : {len(cdA):>4} days")
    print(f"    Window B : {len(cdB):>4} days")
    print(f"    Window C : {len(cdC):>4} days")

    print(f"\n[Step 6] PCP-implied SPX + q calibration ...")
    spx_pcp = compute_pcp_implied_spot(pcp)
    q_imp = compute_q_implied(pcp)
    spx_pcp.to_parquet(cfg.PROCESSED_DIR / 'spx_pcp.parquet')
    q_imp.to_parquet(cfg.PROCESSED_DIR / 'q_implied.parquet')
    print(f"  spx_pcp     : {len(spx_pcp)} dates, mean S = {spx_pcp['S_pcp'].mean():.2f}")
    if len(q_imp):
        print(f"  q_implied   : {len(q_imp)} dates, median q = {q_imp['q_implied'].median():.4f}")
    else:
        print(f"  q_implied   : EMPTY (insufficient cross-maturity pairs)")

    print(f"\n[Step 7] Sanity checks ...")
    results = sanity_checks(panel, meta, qr, q_imp, spx_pcp)
    ok = print_sanity_results(results, halt_on_fail=False)

    if ok:
        print(f"\n  >>> All checks passed.")
    else:
        print(f"\n  >>> Some checks failed. See above. Investigate before next step.")

    print(f"\nOutputs in: {cfg.PROCESSED_DIR}")
    return ok, panel, meta, qr, q_imp, spx_pcp

if __name__ == '__main__':
    main()
