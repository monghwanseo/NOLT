from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.loader_pc1 import build_pc1_bundle_for_fold
from src.models.nolt import NOLT, NOLTConfig
from src.training.trainer import (set_deterministic, train as train_dl,
                                    predict_proba, TrainConfig)

SEED = 2026
OUT_DIR = ROOT / "results"
FOLDS = [3, 4, 5]
FOLD_INDICES = {3: (147, 167, 207), 4: (187, 207, 247), 5: (227, 247, 287)}
THRESHOLD_QUANTILE = 0.85
LOOKBACKS = [1, 30, 60, 96, 120]
ARCH = {"d_model": 128, "n_layers": 2, "dropout": 0.2}

def auc(y, s):
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, s))

def main():
    t0 = time.time()
    print("=" * 70)
    print(f"Phase 2 #7 - Lookback sweep (NOLT, d=128, L=2, dr=0.2, seed=2026)")
    print(f"Sweep: {LOOKBACKS}")
    print("=" * 70)

    out = {"lookbacks": LOOKBACKS, "arch": ARCH, "seed": SEED, "results": {}}
    for lb in LOOKBACKS:
        print(f"\n[lookback={lb}]")
        out["results"][lb] = {}
        for k in FOLDS:
            tr_end, va_end, te_end = FOLD_INDICES[k]
            b = build_pc1_bundle_for_fold(tr_end, va_end, te_end, lookback=lb,
                                            horizon=1, threshold_quantile=THRESHOLD_QUANTILE,
                                            seed=SEED)
            set_deterministic(SEED)
            try:
                m = NOLT(NOLTConfig(n_options=b.n_options, lookback=lb, n_features=1,
                                      d_model=ARCH["d_model"], n_heads=4,
                                      n_layers=ARCH["n_layers"], dropout=ARCH["dropout"]))
                info = train_dl(m, b.X_train, b.y_train, b.X_val, b.y_val,
                                  TrainConfig(epochs=120, batch_size=32, lr=3e-4,
                                                weight_decay=1e-3, early_stop_patience=20,
                                                seed=SEED))
                v = float(info["best_val_auc"])
                t = auc(b.y_test, predict_proba(m, b.X_test))
                out["results"][lb][k] = {"val_auc": v, "test_auc": t,
                                           "best_epoch": info["best_epoch"]}
                print(f"  Fold {k}: val={v:.4f}, test={t:.4f}, "
                      f"best_epoch={info['best_epoch']}")
            except Exception as e:
                print(f"  Fold {k}: FAIL {type(e).__name__}: {e}")
                out["results"][lb][k] = {"error": str(e)}

    summary = {}
    for lb in LOOKBACKS:
        tests = [v["test_auc"] for v in out["results"][lb].values()
                 if "test_auc" in v and not np.isnan(v["test_auc"])]
        vals = [v["val_auc"] for v in out["results"][lb].values()
                if "val_auc" in v and not np.isnan(v["val_auc"])]
        if tests:
            summary[lb] = {
                "median_test": float(np.median(tests)),
                "mean_val": float(np.mean(vals)),
                "per_fold_test": {k: out["results"][lb][k].get("test_auc")
                                   for k in FOLDS},
            }
    out["summary"] = summary

    print("\n" + "=" * 70)
    print("LOOKBACK SWEEP SUMMARY (median test AUC)")
    print("=" * 70)
    for lb in LOOKBACKS:
        if lb in summary:
            pf = summary[lb]["per_fold_test"]
            print(f"  lookback={lb:>4}: median={summary[lb]['median_test']:.4f} "
                  f"(F3={pf[3]:.4f}, F4={pf[4]:.4f}, F5={pf[5]:.4f})")

    out["elapsed_seconds"] = time.time() - t0
    out_path = OUT_DIR / "phase2_lookback_sweep.json"
    out_path.write_text(json.dumps(out, indent=2,
                                      default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_path}  ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
