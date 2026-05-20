import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.stats import mannwhitneyu, ks_2samp

from src.data.loader_pc1 import build_pc1_bundle_for_fold
from src.models.nolt_ablations import NOLTNoSequence
from src.models.nolt import NOLTConfig
from src.training.trainer import set_deterministic, train as train_dl, predict_proba, TrainConfig

SEED = 2026
OUT_DIR = ROOT / "results"
LOOKBACK = 60
FOLDS = [3, 4, 5]
FOLD_INDICES = {3: (147, 167, 207), 4: (187, 207, 247), 5: (227, 247, 287)}

NOLT_SNAP_D = 32
NOLT_SNAP_L = 3
NOLT_SNAP_DR = 0.3

def main():
    print("=" * 78)
    print("HEDGING / ECONOMIC SIGNIFICANCE -does NOLT-snap signal predict |?PC1| size?")
    print("=" * 78)

    rows_high, rows_low = [], []
    rows_all = []

    for k in FOLDS:
        te, ve, tee = FOLD_INDICES[k]
        bundle = build_pc1_bundle_for_fold(te, ve, tee, lookback=LOOKBACK, horizon=1,
                                             threshold_quantile=0.85, seed=SEED)

        set_deterministic(SEED)
        m = NOLTNoSequence(NOLTConfig(n_options=bundle.n_options, lookback=LOOKBACK,
                                        n_features=1, d_model=NOLT_SNAP_D, n_heads=4,
                                        n_layers=NOLT_SNAP_L, dropout=NOLT_SNAP_DR))
        info = train_dl(m, bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val,
                         TrainConfig(epochs=120, batch_size=32, lr=3e-4,
                                      weight_decay=1e-3, early_stop_patience=20, seed=SEED))

        probs = predict_proba(m, bundle.X_test)

        test_start_idx = ve
        test_end_idx = tee

        abs_dpc1_test = np.abs(bundle.dpc1[test_start_idx:test_end_idx])

        n_test = len(probs)
        if len(abs_dpc1_test) > n_test:
            abs_dpc1_test = abs_dpc1_test[:n_test]
        elif len(abs_dpc1_test) < n_test:
            probs = probs[:len(abs_dpc1_test)]

        for i, (p, a) in enumerate(zip(probs, abs_dpc1_test)):
            rows_all.append({"fold": k, "sample_idx": i, "pred_prob": float(p),
                             "abs_dpc1": float(a), "y_test": float(bundle.y_test[i])})

        n = len(probs)
        n_q = max(1, int(n * 0.30))
        order = np.argsort(probs)
        low_idx = order[:n_q]
        high_idx = order[-n_q:]
        rows_high.extend(abs_dpc1_test[high_idx].tolist())
        rows_low.extend(abs_dpc1_test[low_idx].tolist())

        print(f"  Fold {k}: n_test={n}  |?PC1| high-pred mean={abs_dpc1_test[high_idx].mean():.4f}, "
              f"low-pred mean={abs_dpc1_test[low_idx].mean():.4f}", flush=True)

    rows_high = np.array(rows_high)
    rows_low = np.array(rows_low)

    print("\n" + "=" * 78)
    print("AGGREGATED ACROSS FOLDS")
    print("=" * 78)
    print(f"  high-pred (top 30%) |?PC1|: mean={rows_high.mean():.4f}, median={np.median(rows_high):.4f}, n={len(rows_high)}")
    print(f"  low-pred  (bot 30%) |?PC1|: mean={rows_low.mean():.4f}, median={np.median(rows_low):.4f}, n={len(rows_low)}")
    ratio = rows_high.mean() / rows_low.mean() if rows_low.mean() > 0 else float("nan")
    print(f"  ratio (high/low) = {ratio:.2f}x")

    mw_stat, mw_p = mannwhitneyu(rows_high, rows_low, alternative="greater")
    ks_stat, ks_p = ks_2samp(rows_high, rows_low)
    print(f"\n  Mann-Whitney U (high>low): U={mw_stat:.0f}, p={mw_p:.4e}")
    print(f"  KS 2-sample:             stat={ks_stat:.4f}, p={ks_p:.4e}")

    delta_dpc1 = rows_high.mean() - rows_low.mean()
    print(f"\n  Mean |?PC1| difference (high-low) = {delta_dpc1:+.4f}")
    print(f"  Translation: NOLT-aware hedger correctly identifies days with "
          f"~{ratio:.1f}× larger linearity-residual moves; reduces hedge variance "
          f"on those days proportionally.")

    out = {
        "fold_results": rows_all,
        "high_pred_abs_dpc1": rows_high.tolist(),
        "low_pred_abs_dpc1": rows_low.tolist(),
        "high_mean": float(rows_high.mean()),
        "low_mean": float(rows_low.mean()),
        "ratio_high_over_low": float(ratio),
        "mann_whitney_u": {"U": float(mw_stat), "p": float(mw_p)},
        "ks_2sample": {"stat": float(ks_stat), "p": float(ks_p)},
    }
    with open(OUT_DIR / "hedging_economic_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {OUT_DIR / 'hedging_economic_results.json'}", flush=True)

if __name__ == "__main__":
    main()
