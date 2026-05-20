from __future__ import annotations
import json, sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

SEED = 2026
np.random.seed(SEED)
RES = ROOT / "results"

FOMC_DATES = [
    date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
]

CPI_DATES = [
    date(2024, 12, 11),
    date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
    date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
    date(2025, 7, 15), date(2025, 8, 12), date(2025, 9, 11),
    date(2025, 10, 15), date(2025, 11, 13), date(2025, 12, 10),
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 14),
]

NFP_DATES = [
    date(2024, 12, 6),
    date(2025, 1, 10), date(2025, 2, 7), date(2025, 3, 7),
    date(2025, 4, 4), date(2025, 5, 2), date(2025, 6, 6),
    date(2025, 7, 3), date(2025, 8, 1), date(2025, 9, 5),
    date(2025, 10, 3), date(2025, 11, 7), date(2025, 12, 5),
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3),
]

GDP_DATES = [
    date(2025, 1, 30), date(2025, 4, 30), date(2025, 7, 30),
    date(2025, 10, 30), date(2026, 1, 29), date(2026, 4, 29),
]

EARNINGS_WINDOWS = [
    (date(2025, 1, 13), date(2025, 2, 14)),
    (date(2025, 4, 14), date(2025, 5, 15)),
    (date(2025, 7, 14), date(2025, 8, 15)),
    (date(2025, 10, 13), date(2025, 11, 14)),
    (date(2026, 1, 12), date(2026, 2, 13)),
    (date(2026, 4, 13), date(2026, 4, 29)),
]

def expand_event(events_list, half_window_days):
    out = set()
    for e in events_list:
        for offset in range(-half_window_days, half_window_days + 1):
            out.add(e + timedelta(days=offset))
    return out

def opex_dates(start: date, end: date) -> list[date]:
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:

        d = date(y, m, 1)
        while d.weekday() != 4:
            d += timedelta(days=1)
        third_fri = d + timedelta(days=14)
        if start <= third_fri <= end:
            out.append(third_fri)
        m += 1
        if m > 12: m = 1; y += 1
    return out

def main():

    vb = pd.read_parquet(ROOT / "data" / "processed" / "vol_benchmarks.parquet")
    vb.index = pd.to_datetime(vb.index).normalize()
    pc1 = vb["PC1"]
    abs_dpc1 = pc1.diff().abs().dropna()
    abs_dpc1.index = pd.to_datetime(abs_dpc1.index).normalize()
    win_start = abs_dpc1.index.min().date()
    win_end = abs_dpc1.index.max().date()
    print(f"|ΔPC1| series: {win_start} to {win_end}, n={len(abs_dpc1)}")

    fomc_set = expand_event([d for d in FOMC_DATES if win_start <= d <= win_end], 3)
    cpi_set = expand_event([d for d in CPI_DATES if win_start <= d <= win_end], 1)
    nfp_set = expand_event([d for d in NFP_DATES if win_start <= d <= win_end], 1)
    gdp_set = expand_event([d for d in GDP_DATES if win_start <= d <= win_end], 1)
    macro_set = cpi_set | nfp_set | gdp_set

    opex_list = opex_dates(win_start, win_end)
    opex_set = expand_event(opex_list, 1)

    earnings_set = set()
    for s, e in EARNINGS_WINDOWS:
        s = max(s, win_start); e = min(e, win_end)
        if s > e: continue
        d = s
        while d <= e:
            earnings_set.add(d)
            d += timedelta(days=1)

    all_event_set = fomc_set | macro_set | opex_set | earnings_set

    idx_dates = [d.date() for d in abs_dpc1.index]

    def slice_by(event_set):
        mask = np.array([d in event_set for d in idx_dates])
        return abs_dpc1.values[mask], abs_dpc1.values[~mask], int(mask.sum())

    out = {"seed": SEED,
           "window_start": str(win_start),
           "window_end": str(win_end),
           "n_total_days": int(len(abs_dpc1)),
           "n_fomc_events": len([d for d in FOMC_DATES if win_start <= d <= win_end]),
           "n_opex_events": len(opex_list),
           "n_macro_events_unique_set": len(macro_set),
           "n_earnings_window_days": len(earnings_set),
           "tests": {}}

    print("\n[T1] All economic events vs non-events")
    yes, no_, n_yes = slice_by(all_event_set)
    n_no = len(abs_dpc1) - n_yes
    print(f"  {n_yes} event-window days, {n_no} non-event days "
          f"(union of FOMC, OPEX, macro, earnings)")
    print(f"  Mean |ΔPC1|  events: {yes.mean():.4f}, non-events: {no_.mean():.4f}, "
          f"ratio: {yes.mean()/no_.mean():.2f}x")
    mw_u, mw_p = mannwhitneyu(yes, no_, alternative="greater")
    ks_s, ks_p = ks_2samp(yes, no_)
    out["tests"]["T1_all_events"] = {
        "n_event_days": n_yes, "n_nonevent_days": n_no,
        "event_mean": float(yes.mean()), "nonevent_mean": float(no_.mean()),
        "event_median": float(np.median(yes)), "nonevent_median": float(np.median(no_)),
        "ratio_mean": float(yes.mean() / no_.mean()),
        "mann_whitney_U": float(mw_u), "mann_whitney_p_one_sided": float(mw_p),
        "ks_stat": float(ks_s), "ks_p": float(ks_p),
    }
    print(f"  Mann-Whitney p (one-sided > ): {mw_p:.4e}")
    print(f"  KS stat: {ks_s:.4f}, p: {ks_p:.4e}")

    print("\n[T2] Per-event-type mean |ΔPC1|")
    out["tests"]["T2_per_type"] = {}
    nonevent_baseline_mean = no_.mean()
    for label, eset in [("FOMC", fomc_set), ("OPEX", opex_set),
                          ("Macro", macro_set), ("Earnings", earnings_set)]:
        yes_t, no_t, n = slice_by(eset)
        ratio = float(yes_t.mean() / no_t.mean())
        mw_u_t, mw_p_t = mannwhitneyu(yes_t, no_t, alternative="greater")
        out["tests"]["T2_per_type"][label] = {
            "n_event_days": n, "n_nonevent_days": int(len(abs_dpc1) - n),
            "event_mean": float(yes_t.mean()),
            "nonevent_mean": float(no_t.mean()),
            "ratio_mean": ratio,
            "mann_whitney_p_one_sided": float(mw_p_t),
        }
        print(f"  {label:<10} n={n:>3}: event mean={yes_t.mean():.4f}, "
              f"non-event mean={no_t.mean():.4f}, ratio={ratio:.2f}x, "
              f"MW p={mw_p_t:.4e}")

    print("\n[T3] Top decile |ΔPC1| concentration in event windows")
    n_total = len(abs_dpc1)
    n_top = max(1, int(np.ceil(n_total * 0.10)))
    top_idx = np.argsort(abs_dpc1.values)[-n_top:]
    top_dates = set(idx_dates[i] for i in top_idx)
    in_event = sum(1 for d in top_dates if d in all_event_set)
    in_event_random = (n_yes / n_total) * n_top
    out["tests"]["T3_top_decile"] = {
        "n_top_decile": n_top,
        "n_top_in_event_window": in_event,
        "fraction_in_event_window": float(in_event / n_top),
        "expected_random": float(n_yes / n_total),
        "concentration_ratio": float((in_event / n_top) / (n_yes / n_total)),
    }
    print(f"  Top decile n={n_top}: {in_event} fall in event window "
          f"({in_event/n_top:.1%}), random expectation {n_yes/n_total:.1%}, "
          f"concentration {((in_event/n_top)/(n_yes/n_total)):.2f}x")

    print("\n  Top decile breakdown by event type:")
    breakdown = {}
    for label, eset in [("FOMC", fomc_set), ("OPEX", opex_set),
                          ("Macro", macro_set), ("Earnings", earnings_set)]:
        n_in = sum(1 for d in top_dates if d in eset)
        breakdown[label] = {"n": int(n_in), "fraction": float(n_in / n_top)}
        print(f"    {label:<10}: {n_in}/{n_top} ({n_in/n_top:.1%})")
    breakdown["NoEvent"] = {
        "n": n_top - in_event,
        "fraction": float((n_top - in_event) / n_top),
    }
    print(f"    {'NoEvent':<10}: {n_top-in_event}/{n_top} ({(n_top-in_event)/n_top:.1%})")
    out["tests"]["T3_top_decile"]["breakdown_by_type"] = breakdown

    out_path = RES / "phase5_event_window.json"
    out_path.write_text(json.dumps(out, indent=2,
                                      default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_path}")

if __name__ == "__main__":
    main()
