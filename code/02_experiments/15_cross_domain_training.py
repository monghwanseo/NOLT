import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.loader_pc1 import build_pc1_bundle_for_fold
from src.models.nolt import NOLT, NOLTConfig
from src.models.baselines.bsm_threshold import BSMThresholdBaseline
from src.models.baselines.xgboost_baseline import XGBoostBaseline, build_xgboost_features
from src.models.baselines.lstm_single import LSTMSingleOption, LSTMConfig
try:
    from src.models.baselines.garch import GARCHBaseline
    HAS_ARCH = True
except ModuleNotFoundError:
    HAS_ARCH = False
from src.training.trainer import set_deterministic, train as train_dl, predict_proba, TrainConfig

SEED = 2026
OUT_DIR = ROOT / "results"
LOOKBACK = 60
FOLDS = [3, 4, 5]
FOLD_INDICES = {3: (147, 167, 207), 4: (187, 207, 247), 5: (227, 247, 287)}
THRESHOLD_QUANTILE = 0.85

def auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def get_bundles():
    return {k: build_pc1_bundle_for_fold(*FOLD_INDICES[k], lookback=LOOKBACK,
                                           horizon=1, threshold_quantile=THRESHOLD_QUANTILE,
                                           seed=SEED) for k in FOLDS}

def aggregate_select(per_config_per_fold):
    best = None
    for ckey, by_fold in per_config_per_fold.items():
        if not by_fold:
            continue
        agg = float(np.mean([by_fold[k]["val_auc"] for k in by_fold
                              if not np.isnan(by_fold[k]["val_auc"])]))
        if np.isnan(agg):
            continue
        if best is None or agg > best["agg_val"]:
            best = {"config": ckey, "agg_val": agg,
                    "per_fold_test": {k: by_fold[k]["test_auc"] for k in by_fold},
                    "per_fold_val": {k: by_fold[k]["val_auc"] for k in by_fold}}
    if best:
        best["median_test"] = float(np.median(list(best["per_fold_test"].values())))
    return best

def eval_bsm(bundles):
    res = {}
    for tw in [3, 5, 7, 10, 15]:
        ck = f"tw={tw}"; res[ck] = {}
        for k, b in bundles.items():
            try:
                m = BSMThresholdBaseline(tail_window=tw).fit(b.pc1_window_train, b.y_train)
                res[ck][k] = {"val_auc": auc(b.y_val, m.predict_proba(b.pc1_window_val)),
                              "test_auc": auc(b.y_test, m.predict_proba(b.pc1_window_test))}
            except Exception:
                pass
    return res

def eval_garch(bundles):
    if not HAS_ARCH:
        return {}
    res = {}
    for p, q in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        ck = f"({p},{q})"; res[ck] = {}
        for k, b in bundles.items():
            tr_end = FOLD_INDICES[k][0]
            dpc1 = np.diff(b.pc1[:tr_end])
            if len(dpc1) < 30:
                continue
            try:
                g = GARCHBaseline(p=p, q=q).fit(dpc1)
                res[ck][k] = {
                    "val_auc": auc(b.y_val, g.predict_proba(b.pc1_window_val, b.threshold)),
                    "test_auc": auc(b.y_test, g.predict_proba(b.pc1_window_test, b.threshold)),
                }
            except Exception:
                pass
    return res

def eval_xgboost(bundles):
    res = {}
    for n_est, depth, lr in product([30, 50, 100], [3, 5, 7], [0.03, 0.05, 0.1]):
        ck = f"n={n_est},d={depth},lr={lr}"; res[ck] = {}
        for k, b in bundles.items():
            Xf_tr = build_xgboost_features(b.X_train, b.pc1_window_train)
            Xf_v = build_xgboost_features(b.X_val, b.pc1_window_val)
            Xf_te = build_xgboost_features(b.X_test, b.pc1_window_test)
            set_deterministic(SEED)
            try:
                m = XGBoostBaseline(seed=SEED, n_estimators=n_est, max_depth=depth,
                                     learning_rate=lr, reg_lambda=1.0).fit(
                    Xf_tr, b.y_train, Xf_v, b.y_val, early_stopping_rounds=20)
                res[ck][k] = {"val_auc": auc(b.y_val, m.predict_proba(Xf_v)),
                              "test_auc": auc(b.y_test, m.predict_proba(Xf_te))}
            except Exception:
                pass
    return res

def eval_lstm(bundles):
    res = {}
    for hidden, n_l, dr, lr in product([32, 64, 128], [1, 2], [0.1, 0.2, 0.3], [5e-4, 1e-4]):
        ck = f"h={hidden},L={n_l},dr={dr},lr={lr}"; res[ck] = {}
        for k, b in bundles.items():
            set_deterministic(SEED)
            try:
                m = LSTMSingleOption(LSTMConfig(n_options=b.n_options, lookback=LOOKBACK,
                                                  hidden_dim=hidden, n_layers=n_l, dropout=dr))
                info = train_dl(m, b.X_train, b.y_train, b.X_val, b.y_val,
                                 TrainConfig(epochs=120, batch_size=32, lr=lr,
                                              weight_decay=1e-3, early_stop_patience=20, seed=SEED))
                res[ck][k] = {"val_auc": float(info["best_val_auc"]),
                              "test_auc": auc(b.y_test, predict_proba(m, b.X_test)),
                              "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"]}
            except Exception:
                pass
    return res

def eval_nolt(bundles):
    res = {}
    for d, L, dr in product([32, 64, 128], [2, 3], [0.1, 0.2, 0.3]):
        ck = f"d={d},L={L},dr={dr}"; res[ck] = {}
        for k, b in bundles.items():
            set_deterministic(SEED)
            try:
                m = NOLT(NOLTConfig(n_options=b.n_options, lookback=LOOKBACK, n_features=1,
                                     d_model=d, n_heads=4, n_layers=L, dropout=dr))
                info = train_dl(m, b.X_train, b.y_train, b.X_val, b.y_val,
                                 TrainConfig(epochs=120, batch_size=32, lr=3e-4,
                                              weight_decay=1e-3, early_stop_patience=20, seed=SEED))
                res[ck][k] = {"val_auc": float(info["best_val_auc"]),
                              "test_auc": auc(b.y_test, predict_proba(m, b.X_test)),
                              "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"]}
            except Exception:
                pass
    return res

def main():
    t_start = time.time()
    print("=" * 78)
    print("REAL DOMAIN -independent multi-model evaluation (M6c walk-forward)")
    print("seed=2026, train/val/test=70/15/15, agg-val selection")
    print("=" * 78)
    bundles = get_bundles()
    for k, b in bundles.items():
        print(f"  Fold {k}: train={len(b.X_train)} val={len(b.X_val)} test={len(b.X_test)} "
              f"n_pos_test={int(b.y_test.sum())}")

    all_models = {}
    for name, fn in [("bsm_threshold", eval_bsm), ("garch", eval_garch),
                       ("xgboost", eval_xgboost), ("lstm_single", eval_lstm),
                       ("nolt", eval_nolt)]:
        print(f"\n[{name}]"); t = time.time()
        all_models[name] = fn(bundles)
        print(f"  ({time.time()-t:.0f}s) {len(all_models[name])} configs")

    summary = {}
    print("\n" + "=" * 78)
    print("RESULT (aggregate-val selected)")
    print("=" * 78)
    for name, configs in all_models.items():
        best = aggregate_select(configs)
        if best:
            summary[name] = best
            print(f"  {name:>15}: cfg={best['config']:<30}  "
                  f"agg_val={best['agg_val']:.4f}  "
                  f"per_fold_test=" + ",".join(f"F{k}={v:.3f}" for k, v in best["per_fold_test"].items()) +
                  f"  median={best['median_test']:.4f}")

    best_model = max(summary, key=lambda m: summary[m]["median_test"])
    print(f"\nBEST MODEL: {best_model} (median {summary[best_model]['median_test']:.4f})")

    out = {"all_models": all_models, "summary": summary, "best_model": best_model,
            "elapsed_seconds": time.time() - t_start}
    out_path = OUT_DIR / "real_domain_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_path}  ({(time.time()-t_start)/60:.1f}min)")

if __name__ == "__main__":
    main()
