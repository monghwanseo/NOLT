from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

C_NOLT = "#C8102E"
C_FULL = "#34495E"

TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
RES = ROOT / "results"

def loadj(p):
    with open(p) as f: return json.load(f)

def _f4(v):
    if isinstance(v, str): return v
    return f"{float(v):.4f}"

def update_t1_with_synth_full():
    pf = RES / "phase2_synth_nolt_full.json"
    if not pf.exists():
        print("  [skip] Phase 2 synth NOLT-full not done yet")
        return
    data = loadj(pf)
    h_test = data["results"]["heston"]["test_auc"]
    b_test = data["results"]["bates"]["test_auc"]
    print(f"  Synth NOLT-full Heston test = {h_test:.4f}, Bates test = {b_test:.4f}")

    rows = []
    with open(TAB / "T1_cross_domain_auc.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["Model"] == "NOLT with lookback":
            r["Heston test AUC"] = _f4(h_test)
            r["Bates test AUC"] = _f4(b_test)
    fields = ["Model", "Heston test AUC", "Bates test AUC", "Real fold 3",
              "Real fold 4", "Real fold 5", "Real median", "Best config (Real)"]
    with open(TAB / "T1_cross_domain_auc.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  T1 updated with NOLT-with-lookback synth values")

def update_t3_cross_window():
    p1 = loadj(RES / "phase1_robustness.json")
    cw = p1["cross_window"]

    rows = []
    with open(TAB / "T3_phenomenon_stats.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

    def _hl(acf):
        try:
            return f"{-np.log(2)/np.log(acf):.2f}" if 0 < acf < 1 else "-"
        except Exception:
            return "-"

    def make_row(label, src, n_dates, n_options):
        return {
            "Domain": label,
            "PC1 variance share": _f4(src["pc1_var_share"]),
            "PC1 ACF lag 1": _f4(src["pc1_acf_lag1"]),
            "Half life (days)": _hl(src["pc1_acf_lag1"]),
            "PC1 ADF p": _f4(src["pc1_adf_p"]),
            "PC1 KPSS p": _f4(src["pc1_kpss_p"]),
            "Fraction I(1)": "-",
            "Same sign loadings (%)": f"{src['same_sign_loadings_pct']:.1f}",
        }

    new_b = make_row(f"Real Window B ({cw['window_B']['n_dates']}d × {cw['window_B']['n_options']}opt)",
                      cw["window_B"], cw["window_B"]["n_dates"], cw["window_B"]["n_options"])
    new_c = make_row(f"Real Window C ({cw['window_C']['n_dates']}d × {cw['window_C']['n_options']}opt)",
                      cw["window_C"], cw["window_C"]["n_dates"], cw["window_C"]["n_options"])

    out = []
    for r in rows:
        if r["Domain"].startswith("Real Window B") or r["Domain"].startswith("Real Window C"):
            continue
        out.append(r)
        if r["Domain"] == "Real Window A":
            out.append(new_b)
            out.append(new_c)
    with open(TAB / "T3_phenomenon_stats.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)
    print(f"  T3 updated with Windows B and C rows")

def write_t9_vix_family():
    p1 = loadj(RES / "phase1_robustness.json")
    indiv = p1["vix_family_individual_R2"]
    multivariate = p1["vix_family_multivariate"]
    rows = []
    for name, r2 in indiv.items():
        rows.append({"Benchmark": name, "OLS R squared": _f4(r2)})
    rows.append({"Benchmark": "All 7 jointly (multivariate, level)",
                 "OLS R squared": _f4(multivariate["level_R2"])})
    rows.append({"Benchmark": "All 7 jointly (multivariate, first difference)",
                 "OLS R squared": _f4(multivariate["diff_R2"])})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "T9_vix_family.csv", index=False)
    print(f"  T9_vix_family.csv saved ({len(rows)} rows)")

def write_t10_dm_test():
    p1 = loadj(RES / "phase1_robustness.json")
    dm = p1["dm_test"]["comparisons"]
    rows = []
    for name, c in dm.items():
        rows.append({
            "Comparison": f"NOLT vs {name}",
            "Mean diff (NOLT - baseline)": _f4(c["diff_mean"]),
            "Std diff (per fold)": _f4(c["diff_std"]),
            "Diebold-Mariano t": _f4(c["dm_t"]),
            "p-value (one-sided)": _f4(c["p_value_one_sided"]),
            "Holm-adjusted p": _f4(c["p_value_holm_adjusted"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "T10_dm_test.csv", index=False)
    print(f"  T10_dm_test.csv saved ({len(rows)} rows)")

def write_t11_raw_vs_LB():
    p1 = loadj(RES / "phase1_robustness.json")
    rl = p1["raw_vs_LB"]
    lb = rl["L_B_residual"]; raw = rl["raw_delta_gap"]
    rows = [
        {"Statistic": "PC1 variance share", "L_B residual": _f4(lb["pc1_var_share"]),
         "Raw delta-gap": _f4(raw["pc1_var_share"])},
        {"Statistic": "PC1 ACF lag 1", "L_B residual": _f4(lb["pc1_acf_lag1"]),
         "Raw delta-gap": _f4(raw["pc1_acf_lag1"])},
        {"Statistic": "Half life (days)",
         "L_B residual": f"{-np.log(2)/np.log(lb['pc1_acf_lag1']):.2f}",
         "Raw delta-gap": f"{-np.log(2)/np.log(raw['pc1_acf_lag1']):.2f}"},
        {"Statistic": "PC1 ADF p", "L_B residual": _f4(lb["pc1_adf_p"]),
         "Raw delta-gap": _f4(raw["pc1_adf_p"])},
        {"Statistic": "PC1 KPSS p", "L_B residual": _f4(lb["pc1_kpss_p"]),
         "Raw delta-gap": _f4(raw["pc1_kpss_p"])},
        {"Statistic": "Same-sign loadings (%)",
         "L_B residual": f"{lb['same_sign_loadings_pct']:.1f}",
         "Raw delta-gap": f"{raw['same_sign_loadings_pct']:.1f}"},
        {"Statistic": "Pearson(PC1_LB, PC1_raw) (sign-aligned)",
         "L_B residual": _f4(rl["abs_pc1_pearson"]),
         "Raw delta-gap": "-"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "T11_raw_vs_LB.csv", index=False)
    print(f"  T11_raw_vs_LB.csv saved ({len(rows)} rows)")

def write_t12_lookback_sweep():
    pf = RES / "phase2_lookback_sweep.json"
    if not pf.exists():
        print("  [skip] Phase 2 lookback sweep not done yet")
        return
    data = loadj(pf)
    rows = []
    for lb in data["lookbacks"]:
        if str(lb) not in data["summary"] and lb not in data["summary"]:
            continue
        s = data["summary"].get(str(lb), data["summary"].get(lb))
        if not s: continue
        pf_test = s["per_fold_test"]
        rows.append({
            "Lookback (days)": str(lb),
            "Real fold 3": _f4(pf_test[3] if 3 in pf_test else pf_test["3"]),
            "Real fold 4": _f4(pf_test[4] if 4 in pf_test else pf_test["4"]),
            "Real fold 5": _f4(pf_test[5] if 5 in pf_test else pf_test["5"]),
            "Real median": _f4(s["median_test"]),
            "Mean val AUC": _f4(s["mean_val"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "T12_lookback_sweep.csv", index=False)
    print(f"  T12_lookback_sweep.csv saved ({len(rows)} rows)")

def make_lookback_figure():
    pf = RES / "phase2_lookback_sweep.json"
    if not pf.exists():
        print("  [skip] Phase 2 lookback sweep not done yet")
        return
    data = loadj(pf)
    lookbacks = sorted([int(k) for k in data["summary"].keys()])
    medians = [data["summary"][str(lb) if str(lb) in data["summary"] else lb]["median_test"]
               for lb in lookbacks]
    per_fold = {f: [] for f in [3, 4, 5]}
    for lb in lookbacks:
        s = data["summary"][str(lb) if str(lb) in data["summary"] else lb]
        for f in [3, 4, 5]:
            per_fold[f].append(s["per_fold_test"].get(f, s["per_fold_test"].get(str(f))))

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    fold_label = {3: "Early (smallest train)", 4: "Middle", 5: "Late (largest train)"}
    fold_colors = {3: "#888888", 4: "#aaaaaa", 5: "#666666"}
    for f, vals in per_fold.items():
        ax.plot(lookbacks, vals, "o--", color=fold_colors[f], lw=1.0, markersize=5,
                alpha=0.7, label=fold_label[f])

    ax.plot(lookbacks, medians, "o-", color=C_NOLT, lw=2.4, markersize=9,
            markeredgecolor="white", markeredgewidth=1.4, label="Median")

    for lb, v in zip(lookbacks, medians):
        ax.text(lb, v + 0.022, f"{v:.2f}", ha="center", va="bottom",
                color=C_NOLT, fontsize=9.5, fontweight="bold")

    ax.axhline(0.5, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("Temporal lookback (days)")
    ax.set_ylabel("AUC")
    ax.set_xticks(lookbacks)
    ymax = max(max(medians), max(max(v) for v in per_fold.values())) * 1.15
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)

    fig.subplots_adjust(top=0.95, bottom=0.20)
    fig.text(0.5, 0.04,
             "Removing temporal lookback consistently improves NOLT on real SPX.",
             ha="center", va="bottom", fontsize=13)
    pdf_dir = FIG / "pdf"; pdf_dir.mkdir(parents=True, exist_ok=True)
    out_png = FIG / "Figure_07.png"
    out_pdf = pdf_dir / "Figure_07.pdf"
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"  saved: {out_png}")
    print(f"  saved: {out_pdf}")

def main():
    print("=" * 70)
    print("Phase 4 - Integration")
    print("=" * 70)

    print("\n[T1] Update with synth NOLT-full (Phase 2 #7b)")
    update_t1_with_synth_full()

    print("\n[T3] Cross-window phenomenon (Phase 1 #4)")
    update_t3_cross_window()

    print("\n[T9] VIX-family individual R^2 (Phase 1 #3)")
    write_t9_vix_family()

    print("\n[T10] DM test (Phase 1 #8)")
    write_t10_dm_test()

    print("\n[T11] Raw vs L_B PCA (Phase 1 #2)")
    write_t11_raw_vs_LB()

    print("\n[T12] Lookback sweep (Phase 2 #7)")
    write_t12_lookback_sweep()

    print("\n[F-lookback] Lookback sweep figure")
    make_lookback_figure()

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
