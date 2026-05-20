from __future__ import annotations
import json, sys
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
    "font.size": 10, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

C_NOLT = "#C8102E"
C_BSM = "#2CA02C"; C_GARCH = "#FF7F0E"; C_XGB = "#9467BD"; C_LSTM = "#17A2B8"
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
RES = ROOT / "results"

def _f4(v):
    if isinstance(v, str): return v
    return f"{float(v):.4f}"

def write_t13(rp):
    rows = []
    var_label = {"VRP": "VRP", "SKEW": "SKEW", "TS": "TS", "SkewA": "RR"}
    for v in rp["variables"]:
        i_lvl = rp["level"]["individual"][v]; i_dif = rp["diff"]["individual"][v]
        j_lvl = rp["level"]["joint"]; j_dif = rp["diff"]["joint"]
        rows.append({
            "Variable": var_label.get(v, v),
            "Coef (level)": _f4(j_lvl["coefs"][v]),
            "t (level)": _f4(j_lvl["tvalues"][v]),
            "p (level)": _f4(j_lvl["pvalues"][v]),
            "R^2 univariate (level)": _f4(i_lvl["r2"]),
            "Coef (diff)": _f4(j_dif["coefs"][v]),
            "t (diff)": _f4(j_dif["tvalues"][v]),
            "p (diff)": _f4(j_dif["pvalues"][v]),
            "R^2 univariate (diff)": _f4(i_dif["r2"]),
        })
    rows.append({
        "Variable": "Joint OLS R^2",
        "Coef (level)": "—", "t (level)": "—", "p (level)": "—",
        "R^2 univariate (level)": _f4(rp["level"]["joint"]["r2"]),
        "Coef (diff)": "—", "t (diff)": "—", "p (diff)": "—",
        "R^2 univariate (diff)": _f4(rp["diff"]["joint"]["r2"]),
    })
    rows.append({
        "Variable": "Joint OLS adj R^2",
        "Coef (level)": "—", "t (level)": "—", "p (level)": "—",
        "R^2 univariate (level)": _f4(rp["level"]["joint"]["adj_r2"]),
        "Coef (diff)": "—", "t (diff)": "—", "p (diff)": "—",
        "R^2 univariate (diff)": _f4(rp["diff"]["joint"]["adj_r2"]),
    })
    rows.append({
        "Variable": "Joint resid ADF p",
        "Coef (level)": "—", "t (level)": "—", "p (level)": "—",
        "R^2 univariate (level)": _f4(rp["level"]["resid_adf_p"]),
        "Coef (diff)": "—", "t (diff)": "—", "p (diff)": "—",
        "R^2 univariate (diff)": _f4(rp["diff"]["resid_adf_p"]),
    })
    rows.append({
        "Variable": "Joint resid KPSS p",
        "Coef (level)": "—", "t (level)": "—", "p (level)": "—",
        "R^2 univariate (level)": _f4(rp["level"]["resid_kpss_p"]),
        "Coef (diff)": "—", "t (diff)": "—", "p (diff)": "—",
        "R^2 univariate (diff)": _f4(rp["diff"]["resid_kpss_p"]),
    })
    df = pd.DataFrame(rows)
    out = TAB / "T13_risk_premium_regression.csv"
    df.to_csv(out, index=False); print(f"  saved: {out}")

def write_t14(ev):
    rows = []
    rows.append({
        "Window": "All economic events (union)",
        "n event days": ev["tests"]["T1_all_events"]["n_event_days"],
        "Event mean abs dPC1": _f4(ev["tests"]["T1_all_events"]["event_mean"]),
        "Non-event mean abs dPC1": _f4(ev["tests"]["T1_all_events"]["nonevent_mean"]),
        "Ratio (event/non)": _f4(ev["tests"]["T1_all_events"]["ratio_mean"]),
        "Mann-Whitney p": f"{float(ev['tests']['T1_all_events']['mann_whitney_p_one_sided']):.2e}",
        "KS p": f"{float(ev['tests']['T1_all_events']['ks_p']):.2e}",
    })
    for label, c in ev["tests"]["T2_per_type"].items():
        rows.append({
            "Window": f"{label} window",
            "n event days": c["n_event_days"],
            "Event mean abs dPC1": _f4(c["event_mean"]),
            "Non-event mean abs dPC1": _f4(c["nonevent_mean"]),
            "Ratio (event/non)": _f4(c["ratio_mean"]),
            "Mann-Whitney p": f"{float(c['mann_whitney_p_one_sided']):.2e}",
            "KS p": "—",
        })
    rows.append({
        "Window": "Top decile abs dPC1 in event windows",
        "n event days": ev["tests"]["T3_top_decile"]["n_top_in_event_window"],
        "Event mean abs dPC1": "—",
        "Non-event mean abs dPC1": "—",
        "Ratio (event/non)": _f4(ev["tests"]["T3_top_decile"]["concentration_ratio"]),
        "Mann-Whitney p": "—", "KS p": "—",
    })
    df = pd.DataFrame(rows)
    out = TAB / "T14_event_window.csv"
    df.to_csv(out, index=False); print(f"  saved: {out}")

def make_f9(rp, ev):

    var_label = {
        "VRP":   "Variance\nRisk Premium",
        "SKEW":  "CBOE\nSKEW Index",
        "TS":    "Term\nStructure",
        "SkewA": "Risk\nReversal",
    }
    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                            top=0.96, bottom=0.22, left=0.07, right=0.98)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    dates = pd.to_datetime(rp["fitted_level_dates"])
    pc1 = np.array(rp["pc1_level_values"])
    fit = np.array(rp["fitted_level_values"])
    r2_lvl = rp["level"]["joint"]["r2"]
    ax_a.plot(dates, pc1, color="#000000", lw=1.3, label="PC1")
    ax_a.plot(dates, fit, color=C_NOLT, lw=1.2, ls="--", label="Risk-premium fit")
    ax_a.axhline(0, color="grey", lw=0.5, ls="--")

    ymin, ymax = float(np.min([pc1.min(), fit.min()])), float(np.max([pc1.max(), fit.max()]))
    pad = (ymax - ymin) * 0.15
    ax_a.set_ylim(ymin - pad, ymax + pad * 2.0)
    ax_a.set_ylabel("First principal component")
    ax_a.set_xlabel("Date\n\n(a) PC1 levels with joint risk-premium fit", labelpad=10)
    ax_a.legend(loc="upper right", frameon=False)

    ax_a.text(0.01, 0.96, f"$R^2$ = {r2_lvl:.3f}", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", color=C_NOLT, va="top")

    variables = rp["variables"]
    labels_b = [var_label[v] for v in variables]
    r2_lvl_indiv = [rp["level"]["individual"][v]["r2"] for v in variables]
    r2_dif_indiv = [rp["diff"]["individual"][v]["r2"] for v in variables]
    x = np.arange(len(variables)); w = 0.38
    bars1 = ax_b.bar(x - w/2, r2_lvl_indiv, w, color="#1F77B4",
                       edgecolor="black", linewidth=0.5, label="Level")
    bars2 = ax_b.bar(x + w/2, r2_dif_indiv, w, color="#D55E00",
                       edgecolor="black", linewidth=0.5, label="First difference")

    for b, v in zip(bars1, r2_lvl_indiv):
        ax_b.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}",
                   ha="center", fontsize=9)
    for b, v in zip(bars2, r2_dif_indiv):
        ax_b.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}",
                   ha="center", fontsize=9)
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels_b, fontsize=10)
    ax_b.set_ylim(0, max(r2_lvl_indiv + r2_dif_indiv) * 1.85)
    ax_b.set_ylabel("Univariate $R^2$")
    ax_b.set_xlabel("\n(b) Per-variable explanatory power", labelpad=8)

    j_lvl = rp["level"]["joint"]["r2"]; j_dif = rp["diff"]["joint"]["r2"]
    ax_b.text(0.03, 0.96,
               f"Joint level $R^2$ = {j_lvl:.2f}\n"
               f"Joint diff  $R^2$ = {j_dif:.2f}",
               transform=ax_b.transAxes, ha="left", va="top", fontsize=10,
               multialignment="left", color="#222222")

    ax_b.legend(loc="upper right", bbox_to_anchor=(0.97, 1.00),
                frameon=False, fontsize=9.5)

    types = ["FOMC", "OPEX", "Macro", "Earnings"]
    event_means = [ev["tests"]["T2_per_type"][t]["event_mean"] for t in types]
    nonevent_means = [ev["tests"]["T2_per_type"][t]["nonevent_mean"] for t in types]
    ratios = [ev["tests"]["T2_per_type"][t]["ratio_mean"] for t in types]

    x2 = np.arange(len(types)); w2 = 0.38
    b1 = ax_c.bar(x2 - w2/2, event_means, w2, color=C_NOLT,
                   edgecolor="black", linewidth=0.5, label="Event window")
    b2 = ax_c.bar(x2 + w2/2, nonevent_means, w2, color="#888888",
                   edgecolor="black", linewidth=0.5, label="Non-event")

    for b, v in zip(b1, event_means):
        ax_c.text(b.get_x() + b.get_width()/2, v + 0.0006, f"{v:.3f}",
                   ha="center", fontsize=8.5)
    for b, v in zip(b2, nonevent_means):
        ax_c.text(b.get_x() + b.get_width()/2, v + 0.0006, f"{v:.3f}",
                   ha="center", fontsize=8.5)

    max_bar = max(event_means + nonevent_means)
    ymax_c = max_bar * 1.95
    ratio_y = max_bar * 1.22
    for i, r in enumerate(ratios):
        ax_c.text(i, ratio_y, f"{r:.2f}x",
                   ha="center", va="center", fontsize=10.5, color="#222222",
                   fontweight="bold")
    ax_c.set_xticks(x2); ax_c.set_xticklabels(types, fontsize=10)
    ax_c.set_ylabel("Mean $|\\Delta\\mathrm{PC1}|$")
    ax_c.set_xlabel("\n(c) Event window vs non-event mean", labelpad=8)
    ax_c.set_ylim(0, ymax_c)

    ax_c.legend(loc="upper right", bbox_to_anchor=(0.97, 1.00),
                frameon=False, fontsize=9.5)

    fig.text(0.5, 0.04,
              "Risk premia partially explain PC1 levels but transitions remain novel and event independent.",
              ha="center", va="bottom", fontsize=13)
    pdf_dir = FIG / "pdf"; pdf_dir.mkdir(parents=True, exist_ok=True)
    out_png = FIG / "Figure_04.png"
    out_pdf = pdf_dir / "Figure_04.pdf"
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"  saved: {out_png}")
    print(f"  saved: {out_pdf}")

def main():
    print("=" * 70)
    print("Phase 5C - F9 + T13 + T14 generation")
    print("=" * 70)
    rp = json.loads((RES / "phase5_risk_premium.json").read_text())
    ev = json.loads((RES / "phase5_event_window.json").read_text())

    print("\n[T13] Risk premium regression")
    write_t13(rp)

    print("\n[T14] Event window")
    write_t14(ev)

    print("\n[F9] PC1 mechanism (3-panel)")
    make_f9(rp, ev)

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
