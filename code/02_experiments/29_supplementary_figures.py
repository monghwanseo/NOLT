import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

OUT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "ablation_results.json") as f:
    abl = json.load(f)
labels_abl = ["NOLT\nlinear", "NOLT-no-CS", "NOLT-full", "NOLT-snap"]
keys_abl = ["nolt_linear", "nolt_no_crosssection", "nolt_full", "nolt_no_sequence"]
medians = [abl["summary"][k]["median_test"] for k in keys_abl]
colors_abl = ["#aaa", "#88c", "#77b", "black"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
bars = ax.bar(labels_abl, medians, color=colors_abl, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, m in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width()/2, m + 0.01, f"{m:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, label="random")
ax.set_ylabel("Median Test AUROC (Window A, 3 folds)")
ax.set_title("Ablation — NOLT components on real Window A\n"
             "Cross-section attention edge: +0.090; Lookback adverse on small-N: -0.144")
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis="y")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
out = FIG_DIR / "ablation_components.png"
fig.savefig(out); plt.close(fig)
print(f"saved: {out}")

with open(OUT_DIR / "hedging_economic_results.json") as f:
    hed = json.load(f)
high = np.array(hed["high_pred_abs_dpc1"])
low = np.array(hed["low_pred_abs_dpc1"])
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
bins = np.linspace(0, max(high.max(), low.max()) * 1.05, 25)
ax.hist(low, bins=bins, alpha=0.6, color="C0", edgecolor="white",
        label=f"low-pred bottom 30% (n={len(low)}, mean={low.mean():.4f})")
ax.hist(high, bins=bins, alpha=0.6, color="C3", edgecolor="white",
        label=f"high-pred top 30%   (n={len(high)}, mean={high.mean():.4f})")
ax.axvline(low.mean(), color="C0", linestyle="--", linewidth=1.2)
ax.axvline(high.mean(), color="C3", linestyle="--", linewidth=1.2)
ax.set_xlabel(r"|$\Delta$PC1| (actual transition magnitude on test day)")
ax.set_ylabel("count")
mw_p = hed["mann_whitney_u"]["p"]
ax.set_title(f"NOLT-snap predictions correctly identify large-transition days\n"
             f"high/low ratio = {hed['ratio_high_over_low']:.2f}× | "
             f"Mann-Whitney p = {mw_p:.1e}")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
out = FIG_DIR / "hedging_economic.png"
fig.savefig(out); plt.close(fig)
print(f"saved: {out}")
