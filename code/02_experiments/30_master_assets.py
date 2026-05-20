import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"

mpl.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

with open(RESULTS / "real_domain_results.json") as f: real = json.load(f)
with open(RESULTS / "ablation_results.json") as f: abl = json.load(f)
with open(RESULTS / "hedging_economic_results.json") as f: hed = json.load(f)

def fig1_headline():
    rs = real["summary"]; abl_s = abl["summary"]
    snap_real = abl_s["nolt_no_sequence"]["median_test"]
    rows = [
        ("BSM", rs["bsm_threshold"]["median_test"]),
        ("GARCH(2,1)", rs["garch"]["median_test"]),
        ("XGBoost", rs["xgboost"]["median_test"]),
        ("LSTM-single", rs["lstm_single"]["median_test"]),
        ("NOLT-full", rs["nolt"]["median_test"]),
        ("NOLT-snap (ours)", snap_real),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["#7a8b8b", "#5a6a7a", "#aab8c8", "#4a6a8a", "#2a4a6a", "#c8102e"]
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows],
                    color=cols, edgecolor="white", linewidth=0.8)
    for b, (_, v) in zip(bars, rows):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Median Test AUROC (Window A, 3 folds)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Window A — NOLT-snap (cross-section attention) "
                  "captures the phenomenon\n"
                  "vs LSTM-single (single-option DL)", loc="left", pad=12)
    fig.savefig(FIG_DIR / "F1_headline_real_domain.png")
    plt.close(fig)
    print("saved: F1_headline_real_domain")

def fig3_ablation():
    s = abl["summary"]
    order = ["nolt_linear", "nolt_no_crosssection", "nolt_full", "nolt_no_sequence"]
    labels = ["Linear\n(flatten)", "No cross-section\nattention",
              "NOLT-full\n(cross-attn + lookback)", "NOLT-snap\n(cross-attn, no lookback)"]
    vals = [s[k]["median_test"] for k in order]
    cols = ["#cccccc", "#aab8c8", "#4a6a8a", "#c8102e"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.annotate("", xy=(2, 0.652), xytext=(1, 0.562),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
    ax.text(1.5, 0.62, "+0.090\ncross-section\nattention", fontsize=9, ha="center", color="green")
    ax.annotate("", xy=(3, 0.795), xytext=(2, 0.652),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))
    ax.text(2.5, 0.74, "+0.144\nremoving\nlookback", fontsize=9, ha="center", color="darkred")
    ax.set_ylabel("Median Test AUROC (Window A)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation: cross-section attention contributes +0.09;\n"
                  "removing temporal lookback adds +0.14 (small-N regime)",
                  loc="left", pad=12)
    fig.savefig(FIG_DIR / "F3_ablation.png")
    plt.close(fig)
    print("saved: F3_ablation")

def fig5_economic():
    high = np.array(hed["high_pred_abs_dpc1"])
    low = np.array(hed["low_pred_abs_dpc1"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, max(high.max(), low.max()) * 1.05, 22)
    ax.hist(low, bins=bins, alpha=0.65, color="#7a8b8b", edgecolor="white",
            label=f"Low-pred (bottom 30%)\nmean = {low.mean():.4f}, n = {len(low)}")
    ax.hist(high, bins=bins, alpha=0.65, color="#c8102e", edgecolor="white",
            label=f"High-pred (top 30%)\nmean = {high.mean():.4f}, n = {len(high)}")
    ax.axvline(low.mean(), color="#7a8b8b", linestyle="--", linewidth=1.5)
    ax.axvline(high.mean(), color="#c8102e", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"Actual $|\Delta\mathrm{PC1}|$ (transition magnitude on test day)")
    ax.set_ylabel("Frequency")
    ratio = hed['ratio_high_over_low']
    p = hed['mann_whitney_u']['p']
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"NOLT-snap predictions identify large-transition days: {ratio:.2f}x ratio\n"
                  f"Mann-Whitney p = {p:.1e} (***)", loc="left", pad=12)
    fig.savefig(FIG_DIR / "F5_economic_significance.png")
    plt.close(fig)
    print("saved: F5_economic_significance")

def table1_real():
    rs = real["summary"]; abl_s = abl["summary"]
    rows = []
    for name, key, src in [
        ("BSM threshold", "bsm_threshold", rs),
        ("GARCH(2,1)", "garch", rs),
        ("XGBoost", "xgboost", rs),
        ("LSTM-single", "lstm_single", rs),
        ("NOLT-full", "nolt", rs),
        ("NOLT-snap (ours)", "nolt_no_sequence", abl_s),
    ]:
        d = src[key]; pf = d["per_fold_test"]
        rows.append({
            "Model": name,
            "Best config": d["config"],
            "Fold 3": f"{pf.get('3', np.nan):.3f}",
            "Fold 4": f"{pf.get('4', np.nan):.3f}",
            "Fold 5": f"{pf.get('5', np.nan):.3f}",
            "Median": f"{d['median_test']:.4f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "T1_real_domain.csv", index=False)
    print("saved: T1_real_domain.csv")

def table3_ablation():
    s = abl["summary"]
    rows = []
    for label, k in [
        ("Linear (flatten input)", "nolt_linear"),
        ("No cross-section attn (per-option MLP + mean pool)", "nolt_no_crosssection"),
        ("NOLT-full (cross-attn + 60d lookback)", "nolt_full"),
        ("NOLT-snap (cross-attn, no lookback) — ours", "nolt_no_sequence"),
    ]:
        d = s[k]; pf = d["per_fold_test"]
        rows.append({
            "Variant": label,
            "Best config": d["config"],
            "F3": f"{pf.get('3', np.nan):.3f}",
            "F4": f"{pf.get('4', np.nan):.3f}",
            "F5": f"{pf.get('5', np.nan):.3f}",
            "Median": f"{d['median_test']:.4f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "T3_ablation.csv", index=False)
    print("saved: T3_ablation.csv")

def table5_economic():
    high = np.array(hed["high_pred_abs_dpc1"])
    low = np.array(hed["low_pred_abs_dpc1"])
    rows = [
        {"Quantile": "Top 30% (high-pred)", "n": len(high),
         "mean |dPC1|": f"{high.mean():.4f}", "median |dPC1|": f"{np.median(high):.4f}"},
        {"Quantile": "Bottom 30% (low-pred)", "n": len(low),
         "mean |dPC1|": f"{low.mean():.4f}", "median |dPC1|": f"{np.median(low):.4f}"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "T5_economic.csv", index=False)
    print("saved: T5_economic.csv")

if __name__ == "__main__":
    print("Generating paper assets (PNG only; PDFs generated on demand later)...\n")
    fig1_headline()
    fig3_ablation()
    fig5_economic()
    print()
    table1_real()
    table3_ablation()
    table5_economic()
    print("\nAll paper assets in paper/figures/ and paper/tables/")
