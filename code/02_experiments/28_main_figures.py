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

with open(OUT_DIR / "real_domain_results.json") as f:
    real = json.load(f)

REAL_MODELS = ["bsm_threshold", "garch", "xgboost", "lstm_single", "nolt"]
LABELS = {"bsm_threshold": "BSM", "garch": "GARCH", "xgboost": "XGBoost",
          "lstm_single": "LSTM", "nolt": "NOLT-full"}
COLORS = {"bsm_threshold": "C0", "garch": "C1", "xgboost": "C2",
          "lstm_single": "C3", "nolt": "black"}

real_summary = real["summary"]
folds = ["3", "4", "5"]
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
width = 0.15
x = np.arange(len(folds))
for i, m in enumerate(REAL_MODELS):
    if m not in real_summary:
        continue
    pft = real_summary[m]["per_fold_test"]
    vals = [pft.get(f, np.nan) for f in folds]
    med = real_summary[m]["median_test"]
    ax.bar(x + i*width, vals, width, label=f"{LABELS[m]} (med={med:.3f})",
            color=COLORS[m], alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, label="random")
ax.set_xticks(x + (len(REAL_MODELS) - 1) * width / 2)
ax.set_xticklabels([f"Fold {f}" for f in folds])
ax.set_ylabel("Test AUROC")
ax.set_title("Real Window A — per-fold model comparison (M6c walk-forward)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.0)
fig.tight_layout()
out = FIG_DIR / "real_domain_comparison.png"
fig.savefig(out)
plt.close(fig)
print(f"saved: {out}")
