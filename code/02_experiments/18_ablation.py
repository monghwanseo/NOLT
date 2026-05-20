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
from src.models.nolt_ablations import NOLTNoCrossSection, NOLTNoSequence, NOLTLinear
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

def eval_variant(name, model_class, bundles):
    res = {}
    grid = list(product([32, 64, 128], [2, 3], [0.1, 0.2, 0.3]))
    print(f"\n[{name}] sweep {len(grid)} configs × {len(bundles)} folds = {len(grid)*len(bundles)} trainings", flush=True)
    for d, L, dr in grid:
        ck = f"d={d},L={L},dr={dr}"
        res[ck] = {}
        for k, b in bundles.items():
            set_deterministic(SEED)
            try:
                m = model_class(NOLTConfig(
                    n_options=b.n_options, lookback=LOOKBACK, n_features=1,
                    d_model=d, n_heads=4, n_layers=L, dropout=dr,
                ))
                info = train_dl(m, b.X_train, b.y_train, b.X_val, b.y_val,
                                 TrainConfig(epochs=120, batch_size=32, lr=3e-4,
                                              weight_decay=1e-3, early_stop_patience=20, seed=SEED))
                t = auc(b.y_test, predict_proba(m, b.X_test))
                res[ck][k] = {"val_auc": float(info["best_val_auc"]),
                              "test_auc": t,
                              "best_epoch": info["best_epoch"],
                              "epochs_run": info["epochs_run"]}
            except Exception as e:
                print(f"    [{name} {ck} fold {k}] failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return res

def main():
    t_start = time.time()
    print("=" * 78)
    print("ABLATION -NOLT component-level edge on REAL Window A")
    print("seed=2026, M6c walk-forward, agg-val selection")
    print("=" * 78)

    bundles = get_bundles()
    print()
    for k, b in bundles.items():
        print(f"  Fold {k}: train={len(b.X_train)} val={len(b.X_val)} test={len(b.X_test)} n_pos_test={int(b.y_test.sum())}", flush=True)

    variants = [
        ("nolt_full", NOLT),
        ("nolt_no_crosssection", NOLTNoCrossSection),
        ("nolt_no_sequence", NOLTNoSequence),
        ("nolt_linear", NOLTLinear),
    ]

    all_results = {}
    summary = {}
    for name, cls in variants:
        t = time.time()
        all_results[name] = eval_variant(name, cls, bundles)
        best = aggregate_select(all_results[name])
        if best:
            summary[name] = best
            print(f"  ({time.time()-t:.0f}s) BEST: cfg={best['config']:<25}  "
                  f"agg_val={best['agg_val']:.4f}  median_test={best['median_test']:.4f}  "
                  f"per_fold=" + ",".join(f"F{k}={v:.3f}" for k, v in best["per_fold_test"].items()), flush=True)

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY (median test AUROC, agg-val selected)")
    print("=" * 78)
    for name, b in summary.items():
        print(f"  {name:>22}: median = {b['median_test']:.4f}  cfg={b['config']}", flush=True)

    if "nolt_full" in summary and "nolt_no_crosssection" in summary:
        edge = summary["nolt_full"]["median_test"] - summary["nolt_no_crosssection"]["median_test"]
        print(f"\nCross-section attention edge: NOLT-full -NOLT-no-crosssection = {edge:+.4f}", flush=True)
    if "nolt_full" in summary and "nolt_no_sequence" in summary:
        seq_edge = summary["nolt_full"]["median_test"] - summary["nolt_no_sequence"]["median_test"]
        print(f"Sequence (lookback) edge: NOLT-full -NOLT-no-sequence = {seq_edge:+.4f}", flush=True)

    out = {"all_results": all_results, "summary": summary,
            "elapsed_seconds": time.time() - t_start}
    out_path = OUT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_path}  ({(time.time()-t_start)/60:.1f}min)", flush=True)

if __name__ == "__main__":
    main()
