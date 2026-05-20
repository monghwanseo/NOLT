import pandas as pd

from .tasks import common_dates, window_a_tickers, window_b_tickers, window_c_tickers

def sanity_checks(panel, meta, quality_report, q_implied, spx_pcp) -> dict:
    r = {}

    rows_panel = len(panel)
    rows_sum = int(meta['n_rows'].sum())
    r['1_panel_rows_match'] = (
        rows_panel == rows_sum,
        f"panel={rows_panel:,}, sum(meta.n_rows)={rows_sum:,}",
    )

    n_opt = len(meta)
    r['2_n_options_56'] = (n_opt == 56, f"got {n_opt}, expected 56")

    excl = quality_report.loc[quality_report['classification'] == 'EXCLUDE', 'ticker'].tolist()
    r['3_exclude_list'] = (True, f"{len(excl)} excluded: {excl}")

    if len(q_implied) > 0:
        q_med = float(q_implied['q_implied'].median())
        r['4_q_implied_range'] = (
            0.005 <= q_med <= 0.025,
            f"q_implied median = {q_med:.4f} (target ~0.0117)",
        )
    else:
        r['4_q_implied_range'] = (False, "no q_implied data")

    if len(spx_pcp) > 1:
        s = spx_pcp.sort_values('Date')['S_pcp']
        ret = s.pct_change().dropna()
        outlier_rate = float((ret.abs() > 0.10).mean())
        r['5_pcp_outlier_rate'] = (
            outlier_rate < 0.05,
            f"|daily change|>10% rate = {outlier_rate:.4f} (n={len(ret):,})",
        )
    else:
        r['5_pcp_outlier_rate'] = (False, "insufficient PCP data")

    ta = window_a_tickers(quality_report)
    cdA = common_dates(panel, ta)
    r['6_window_a_common'] = (
        len(cdA) >= 200,
        f"Window A common days = {len(cdA)} ({len(ta)} options)",
    )

    tb = window_b_tickers(quality_report)
    cdB = common_dates(panel, tb)
    r['7_window_b_common'] = (
        len(cdB) >= 50,
        f"Window B common days = {len(cdB)} ({len(tb)} options)",
    )

    bad = [c for c in panel.columns if isinstance(c, str) and ('(R' in c or '(L' in c)]
    r['8_col_normalization'] = (len(bad) == 0, f"unnormalized cols: {bad}")

    tc = window_c_tickers(meta, quality_report)
    r['9_window_c_count'] = (len(tc) >= 1, f"Window C options (long history): {len(tc)}")

    return r

def print_sanity_results(results: dict, halt_on_fail: bool = True) -> bool:
    print("=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    n_pass, n_fail = 0, 0
    fails = []
    for name, (ok, msg) in results.items():
        flag = "[PASS]" if ok else "[FAIL]"
        print(f"  {flag} {name:25s} : {msg}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            fails.append(name)
    print(f"\n  Total: {n_pass} passed, {n_fail} failed")
    if n_fail > 0 and halt_on_fail:
        raise RuntimeError(f"Sanity checks failed: {fails}. Halting.")
    return n_fail == 0
