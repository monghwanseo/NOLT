from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.data.loader_pc1 import build_pc1_bundle_for_fold

SEED = 2026
LOOKBACK = 60
FOLDS = [3, 4, 5]
FOLD_INDICES = {3: (147, 167, 207), 4: (187, 207, 247), 5: (227, 247, 287)}
THRESHOLD_QUANTILE = 0.85

def _feat_pc1_last(pc1_win):
    return pc1_win[:, -1]

def _feat_pc1_mean_k(pc1_win, k):
    return pc1_win[:, -k:].mean(axis=1)

def _feat_pc1_std_k(pc1_win, k):
    return pc1_win[:, -k:].std(axis=1)

def _feat_abs_dpc1_max_k(pc1_win, k):
    dpc1 = np.abs(np.diff(pc1_win, axis=1))
    return dpc1[:, -k:].max(axis=1)

def _feat_abs_dpc1_mean_k(pc1_win, k):
    dpc1 = np.abs(np.diff(pc1_win, axis=1))
    return dpc1[:, -k:].mean(axis=1)

def _feat_R_l1_now(X_win):
    return np.abs(X_win[:, -1, :]).sum(axis=1)

def _feat_R_linf_now(X_win):
    return np.abs(X_win[:, -1, :]).max(axis=1)

def _feat_R_l1_mean_k(X_win, k):
    return np.abs(X_win[:, -k:, :]).sum(axis=2).mean(axis=1)

BASELINES: list[tuple[str, Callable]] = [
    ("B1_PC1_t",            lambda b: _feat_pc1_last(b)),
    ("B2_PC1_mean_5",       lambda b: _feat_pc1_mean_k(b, 5)),
    ("B3_PC1_mean_20",      lambda b: _feat_pc1_mean_k(b, 20)),
    ("B4_PC1_std_10",       lambda b: _feat_pc1_std_k(b, 10)),
    ("B5_absdPC1_max_5",    lambda b: _feat_abs_dpc1_max_k(b, 5)),
    ("B6_absdPC1_max_20",   lambda b: _feat_abs_dpc1_max_k(b, 20)),
    ("B7_absdPC1_mean_10",  lambda b: _feat_abs_dpc1_mean_k(b, 10)),
]
BASELINES_R: list[tuple[str, Callable]] = [
    ("B8_R_L1_now",         lambda X: _feat_R_l1_now(X)),
    ("B9_R_Linf_now",       lambda X: _feat_R_linf_now(X)),
    ("B10_R_L1_mean_5",     lambda X: _feat_R_l1_mean_k(X, 5)),
]

def auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def fit_eval(f_train, y_train, f_test, y_test):
    X_tr = f_train.reshape(-1, 1)
    X_te = f_test.reshape(-1, 1)

    mu = X_tr.mean()
    sd = X_tr.std() if X_tr.std() > 1e-12 else 1.0
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    model = LogisticRegression(random_state=SEED, max_iter=1000, solver="lbfgs",
                                C=1.0, class_weight="balanced")
    model.fit(X_tr, y_train.astype(int))
    s_te = model.predict_proba(X_te)[:, 1]
    return auc(y_test, s_te), float(model.coef_[0, 0]), float(model.intercept_[0])

def get_bundles():
    return {k: build_pc1_bundle_for_fold(*FOLD_INDICES[k], lookback=LOOKBACK,
                                           horizon=1,
                                           threshold_quantile=THRESHOLD_QUANTILE,
                                           seed=SEED) for k in FOLDS}

def main():
    t0 = time.time()
    print("=" * 78)
    print("E5 — simple-baselines on PRIMARY residual matrix")
    print(f"seed={SEED}, folds={FOLDS}, threshold_q={THRESHOLD_QUANTILE}, lookback={LOOKBACK}")
    print("=" * 78)
    bundles = get_bundles()
    for k, b in bundles.items():
        print(f"  Fold {k}: train={len(b.X_train)} val={len(b.X_val)} "
              f"test={len(b.X_test)} n_pos_test={int(b.y_test.sum())}")

    rows = []
    all_results = {}

    for name, fn in BASELINES:
        per_fold = {}
        for k, b in bundles.items():
            f_tr = fn(b.pc1_window_train)
            f_te = fn(b.pc1_window_test)
            a, beta, intercept = fit_eval(f_tr, b.y_train, f_te, b.y_test)
            per_fold[k] = {"test_auc": a, "coef": beta, "intercept": intercept}
        aucs = np.array([per_fold[k]["test_auc"] for k in FOLDS])
        med = float(np.median(aucs))
        mean = float(np.mean(aucs))
        rows.append({
            "Baseline": name,
            "Fold 3": per_fold[3]["test_auc"],
            "Fold 4": per_fold[4]["test_auc"],
            "Fold 5": per_fold[5]["test_auc"],
            "Median": med,
            "Mean": mean,
        })
        all_results[name] = per_fold
        print(f"  {name:<25} F3={per_fold[3]['test_auc']:.4f} F4={per_fold[4]['test_auc']:.4f} "
              f"F5={per_fold[5]['test_auc']:.4f}  median={med:.4f}")

    for name, fn in BASELINES_R:
        per_fold = {}
        for k, b in bundles.items():
            f_tr = fn(b.X_train)
            f_te = fn(b.X_test)
            a, beta, intercept = fit_eval(f_tr, b.y_train, f_te, b.y_test)
            per_fold[k] = {"test_auc": a, "coef": beta, "intercept": intercept}
        aucs = np.array([per_fold[k]["test_auc"] for k in FOLDS])
        med = float(np.median(aucs))
        mean = float(np.mean(aucs))
        rows.append({
            "Baseline": name,
            "Fold 3": per_fold[3]["test_auc"],
            "Fold 4": per_fold[4]["test_auc"],
            "Fold 5": per_fold[5]["test_auc"],
            "Median": med,
            "Mean": mean,
        })
        all_results[name] = per_fold
        print(f"  {name:<25} F3={per_fold[3]['test_auc']:.4f} F4={per_fold[4]['test_auc']:.4f} "
              f"F5={per_fold[5]['test_auc']:.4f}  median={med:.4f}")

    out_json = ROOT / "results" / "E5_simple_baselines.json"
    out_json.write_text(json.dumps({
        "seed": SEED, "folds": FOLDS, "threshold_quantile": THRESHOLD_QUANTILE,
        "lookback": LOOKBACK, "residual_matrix": "PRIMARY (Window A, build_residual_matrix)",
        "per_baseline": all_results,
        "summary_table": rows,
    }, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_json}")

    df = pd.DataFrame(rows)
    out_csv = ROOT / "paper" / "tables" / "T_simple_baselines.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"saved: {out_csv}")
    print(f"\nelapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
