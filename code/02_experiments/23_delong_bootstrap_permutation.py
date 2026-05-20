from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

from src.data.loader_pc1 import build_pc1_bundle_for_fold
from src.models.nolt import NOLT, NOLTConfig
from src.models.nolt_ablations import NOLTNoSequence
from src.models.baselines.bsm_threshold import BSMThresholdBaseline
from src.models.baselines.xgboost_baseline import XGBoostBaseline, build_xgboost_features
from src.models.baselines.lstm_single import LSTMSingleOption, LSTMConfig
try:
    from src.models.baselines.garch import GARCHBaseline
    HAS_GARCH = True
except ModuleNotFoundError:
    HAS_GARCH = False
from src.training.trainer import set_deterministic, train as train_dl, predict_proba, TrainConfig

SEED = 2026
LOOKBACK = 60
FOLDS = [3, 4, 5]
FOLD_INDICES = {3: (147, 167, 207), 4: (187, 207, 247), 5: (227, 247, 287)}
THRESHOLD_QUANTILE = 0.85

BEST = {
    "BSM": {"tail_window": 3},
    "GARCH(1,1)": {"p": 1, "q": 1},
    "XGBoost": {"n_estimators": 30, "max_depth": 7, "learning_rate": 0.1},
    "LSTM": {"hidden_dim": 64, "n_layers": 2, "dropout": 0.1, "lr": 1e-4},
    "NOLT_w_lookback": {"d_model": 128, "n_layers": 2, "dropout": 0.2, "lr": 3e-4},
    "NOLT": {"d_model": 32, "n_layers": 3, "dropout": 0.3, "lr": 3e-4},
}

def get_bundles():
    return {k: build_pc1_bundle_for_fold(*FOLD_INDICES[k], lookback=LOOKBACK,
                                           horizon=1,
                                           threshold_quantile=THRESHOLD_QUANTILE,
                                           seed=SEED) for k in FOLDS}

def train_all_and_predict():
    bundles = get_bundles()
    preds: dict[str, dict[int, dict]] = {m: {} for m in BEST}

    for k, b in bundles.items():
        print(f"\n--- Fold {k}: train={len(b.X_train)} val={len(b.X_val)} test={len(b.X_test)} ---")

        cfg_b = BEST["BSM"]
        m_bsm = BSMThresholdBaseline(tail_window=cfg_b["tail_window"]).fit(
            b.pc1_window_train, b.y_train)
        s = m_bsm.predict_proba(b.pc1_window_test)
        preds["BSM"][k] = {"y": b.y_test.tolist(), "s": s.tolist(),
                            "auc": float(roc_auc_score(b.y_test, s)) if len(set(b.y_test)) > 1 else None}
        print(f"  BSM   AUC = {preds['BSM'][k]['auc']:.4f}")

        if HAS_GARCH:
            cfg_g = BEST["GARCH(1,1)"]
            tr_end = FOLD_INDICES[k][0]
            dpc1 = np.diff(b.pc1[:tr_end])
            try:
                gm = GARCHBaseline(p=cfg_g["p"], q=cfg_g["q"]).fit(dpc1)
                s = gm.predict_proba(b.pc1_window_test, b.threshold)
                preds["GARCH(1,1)"][k] = {"y": b.y_test.tolist(), "s": s.tolist(),
                                           "auc": float(roc_auc_score(b.y_test, s))}
                print(f"  GARCH AUC = {preds['GARCH(1,1)'][k]['auc']:.4f}")
            except Exception as e:
                print(f"  GARCH failed: {e}")

        cfg_x = BEST["XGBoost"]
        Xf_tr = build_xgboost_features(b.X_train, b.pc1_window_train)
        Xf_v = build_xgboost_features(b.X_val, b.pc1_window_val)
        Xf_te = build_xgboost_features(b.X_test, b.pc1_window_test)
        set_deterministic(SEED)
        mxg = XGBoostBaseline(seed=SEED, **cfg_x, reg_lambda=1.0).fit(
            Xf_tr, b.y_train, Xf_v, b.y_val, early_stopping_rounds=20)
        s = mxg.predict_proba(Xf_te)
        preds["XGBoost"][k] = {"y": b.y_test.tolist(), "s": s.tolist(),
                                "auc": float(roc_auc_score(b.y_test, s))}
        print(f"  XGB   AUC = {preds['XGBoost'][k]['auc']:.4f}")

        cfg_l = BEST["LSTM"]
        set_deterministic(SEED)
        m_lstm = LSTMSingleOption(LSTMConfig(
            n_options=b.n_options, lookback=LOOKBACK,
            hidden_dim=cfg_l["hidden_dim"], n_layers=cfg_l["n_layers"],
            dropout=cfg_l["dropout"]))
        train_dl(m_lstm, b.X_train, b.y_train, b.X_val, b.y_val,
                  TrainConfig(epochs=120, batch_size=32, lr=cfg_l["lr"],
                              weight_decay=1e-3, early_stop_patience=20, seed=SEED))
        s = predict_proba(m_lstm, b.X_test)
        preds["LSTM"][k] = {"y": b.y_test.tolist(), "s": s.tolist(),
                             "auc": float(roc_auc_score(b.y_test, s))}
        print(f"  LSTM  AUC = {preds['LSTM'][k]['auc']:.4f}")

        cfg_nl = BEST["NOLT_w_lookback"]
        set_deterministic(SEED)
        m_nl = NOLT(NOLTConfig(n_options=b.n_options, lookback=LOOKBACK, n_features=1,
                                 d_model=cfg_nl["d_model"], n_heads=4,
                                 n_layers=cfg_nl["n_layers"], dropout=cfg_nl["dropout"]))
        train_dl(m_nl, b.X_train, b.y_train, b.X_val, b.y_val,
                  TrainConfig(epochs=120, batch_size=32, lr=cfg_nl["lr"],
                              weight_decay=1e-3, early_stop_patience=20, seed=SEED))
        s = predict_proba(m_nl, b.X_test)
        preds["NOLT_w_lookback"][k] = {"y": b.y_test.tolist(), "s": s.tolist(),
                                         "auc": float(roc_auc_score(b.y_test, s))}
        print(f"  NOLT-w-lookback AUC = {preds['NOLT_w_lookback'][k]['auc']:.4f}")

        cfg_n = BEST["NOLT"]
        set_deterministic(SEED)
        m_n = NOLTNoSequence(NOLTConfig(
            n_options=b.n_options, lookback=LOOKBACK, n_features=1,
            d_model=cfg_n["d_model"], n_heads=4,
            n_layers=cfg_n["n_layers"], dropout=cfg_n["dropout"]))
        train_dl(m_n, b.X_train, b.y_train, b.X_val, b.y_val,
                  TrainConfig(epochs=120, batch_size=32, lr=cfg_n["lr"],
                              weight_decay=1e-3, early_stop_patience=20, seed=SEED))
        s = predict_proba(m_n, b.X_test)
        preds["NOLT"][k] = {"y": b.y_test.tolist(), "s": s.tolist(),
                              "auc": float(roc_auc_score(b.y_test, s))}
        print(f"  NOLT-snap AUC = {preds['NOLT'][k]['auc']:.4f}")

    return preds

def _delong_placements(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos = (y == 1); neg = (y == 0)
    s_pos = s[pos]; s_neg = s[neg]
    m = len(s_pos); n = len(s_neg)
    if m == 0 or n == 0:
        return None

    r_all = rankdata(np.concatenate([s_pos, s_neg]), method="average")
    r_pos_in_all = r_all[:m]
    r_neg_in_all = r_all[m:]
    r_pos_self = rankdata(s_pos, method="average")
    r_neg_self = rankdata(s_neg, method="average")

    V10 = (r_pos_in_all - r_pos_self) / n

    V01 = 1.0 - (r_neg_in_all - r_neg_self) / m
    return V10, V01, m, n

def delong_paired(y, s_a, s_b):
    pa = _delong_placements(y, s_a)
    pb = _delong_placements(y, s_b)
    if pa is None or pb is None:
        return None
    V10a, V01a, m, n = pa
    V10b, V01b, _, _ = pb
    auc_a = V10a.mean()
    auc_b = V10b.mean()

    def _cov(x, y):
        if len(x) < 2:
            return 0.0
        return float(np.cov(x, y, ddof=1)[0, 1])
    s10 = _cov(V10a, V10b) / m if m >= 2 else 0.0
    s01 = _cov(V01a, V01b) / n if n >= 2 else 0.0
    var_a = (np.var(V10a, ddof=1) / m if m >= 2 else 0.0) + (np.var(V01a, ddof=1) / n if n >= 2 else 0.0)
    var_b = (np.var(V10b, ddof=1) / m if m >= 2 else 0.0) + (np.var(V01b, ddof=1) / n if n >= 2 else 0.0)
    var_diff = var_a + var_b - 2.0 * (s10 + s01)
    if var_diff <= 0:
        z = float("inf") if (auc_a > auc_b) else float("-inf")
        p = 0.0 if (auc_a > auc_b) else 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        p = float(1.0 - norm.cdf(z))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b),
            "diff": float(auc_a - auc_b),
            "var_diff": float(var_diff),
            "z": float(z), "p_one_sided": float(p)}

def run_delong_per_fold(preds):
    out = {}
    baselines = ["BSM", "GARCH(1,1)", "XGBoost", "LSTM", "NOLT_w_lookback"]
    for k in FOLDS:
        nolt = preds["NOLT"].get(k)
        if nolt is None:
            continue
        y = np.array(nolt["y"], dtype=int)
        s_nolt = np.array(nolt["s"], dtype=float)
        fold_results = {}
        p_values = []
        for bname in baselines:
            if k not in preds.get(bname, {}):
                continue
            s_base = np.array(preds[bname][k]["s"], dtype=float)
            res = delong_paired(y, s_nolt, s_base)
            if res is not None:
                fold_results[bname] = res
                p_values.append(res["p_one_sided"])

        n_t = len(p_values)
        if n_t > 0:
            sorted_idx = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_idx]
            holm_adj = np.zeros(n_t)
            for i in range(n_t):
                holm_adj[i] = min((n_t - i) * sorted_p[i], 1.0)
            for i in range(1, n_t):
                holm_adj[i] = max(holm_adj[i], holm_adj[i - 1])
            holm_back = np.zeros(n_t)
            holm_back[sorted_idx] = holm_adj
            for (bname, res), p_h in zip(fold_results.items(), holm_back):
                res["p_holm"] = float(p_h)
        out[str(k)] = fold_results
    return out

def stationary_block_bootstrap_auc(y, s, B=1000, block_p=1.0 / 7.0, seed=SEED):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    aucs = np.empty(B)
    fail = 0
    for b in range(B):
        idx = np.empty(n, dtype=int)

        i = 0
        start = rng.integers(0, n)
        idx[0] = start
        for t in range(1, n):
            if rng.random() < block_p:

                idx[t] = rng.integers(0, n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        yb = y[idx]; sb = s[idx]
        if len(np.unique(yb)) < 2:
            aucs[b] = np.nan
            fail += 1
        else:
            aucs[b] = roc_auc_score(yb, sb)
    return aucs, fail

def run_bootstrap_per_fold(preds, B=1000):
    out = {}
    rng_base = np.random.default_rng(SEED)
    for m_name, by_fold in preds.items():
        out[m_name] = {}
        for k, d in by_fold.items():
            y = np.array(d["y"], dtype=int)
            s = np.array(d["s"], dtype=float)

            seed_mf = int(rng_base.integers(0, 2**31 - 1))
            aucs, fail = stationary_block_bootstrap_auc(y, s, B=B, seed=seed_mf)
            valid = aucs[~np.isnan(aucs)]
            if len(valid) > 0:
                ci_lo = float(np.percentile(valid, 2.5))
                ci_hi = float(np.percentile(valid, 97.5))
                mean = float(np.mean(valid))
            else:
                ci_lo = ci_hi = mean = float("nan")
            out[m_name][str(k)] = {
                "auc_point": d["auc"],
                "auc_boot_mean": mean,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
                "n_valid_boot": int(len(valid)),
                "n_fail_boot": int(fail),
            }
    return out

def label_permutation_null(y, s, B=10000, seed=SEED):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    nulls = np.empty(B)
    for b in range(B):
        yp = rng.permutation(y)
        if len(np.unique(yp)) < 2:
            nulls[b] = np.nan
        else:
            nulls[b] = roc_auc_score(yp, s)
    return nulls

def run_permutation_per_fold(preds, B=10000):
    out = {}
    rng_base = np.random.default_rng(SEED + 1)
    for m_name, by_fold in preds.items():
        out[m_name] = {}
        for k, d in by_fold.items():
            y = np.array(d["y"], dtype=int)
            s = np.array(d["s"], dtype=float)
            seed_mf = int(rng_base.integers(0, 2**31 - 1))
            nulls = label_permutation_null(y, s, B=B, seed=seed_mf)
            valid = nulls[~np.isnan(nulls)]
            mu = float(np.mean(valid))
            sd = float(np.std(valid, ddof=1))
            actual = d["auc"]
            z = float((actual - mu) / sd) if sd > 0 else float("nan")
            p_one_sided = float(np.mean(valid >= actual))
            out[m_name][str(k)] = {
                "auc_point": actual,
                "null_mean": mu,
                "null_std": sd,
                "z_score": z,
                "p_one_sided": p_one_sided,
                "n_valid": int(len(valid)),
            }
    return out

def save_delong_csv(delong_out):
    rows = []
    for fold in FOLDS:
        per_fold = delong_out.get(str(fold), {})
        for b, r in per_fold.items():
            rows.append({
                "Fold": fold,
                "Comparison": f"NOLT vs {b}",
                "AUC NOLT": r["auc_a"],
                "AUC baseline": r["auc_b"],
                "Diff": r["diff"],
                "DeLong z": r["z"],
                "p-value (one-sided)": r["p_one_sided"],
                "Holm-adjusted p": r.get("p_holm", float("nan")),
            })
    df = pd.DataFrame(rows)
    out_csv = ROOT / "paper" / "tables" / "T10_dm_test.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    return out_csv

def save_bootstrap_csv(boot_out):
    rows = []
    for m_name, by_fold in boot_out.items():
        for k_str, r in by_fold.items():
            rows.append({
                "Model": m_name,
                "Fold": int(k_str),
                "AUC (point)": r["auc_point"],
                "Bootstrap mean AUC": r["auc_boot_mean"],
                "95% CI lo": r["ci_95_lo"],
                "95% CI hi": r["ci_95_hi"],
                "n valid boot": r["n_valid_boot"],
            })
    df = pd.DataFrame(rows).sort_values(["Model", "Fold"])
    out_csv = ROOT / "paper" / "tables" / "T_bootstrap_ci.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    return out_csv

def save_permutation_csv(perm_out):
    rows = []
    for m_name, by_fold in perm_out.items():
        for k_str, r in by_fold.items():
            rows.append({
                "Model": m_name,
                "Fold": int(k_str),
                "AUC (point)": r["auc_point"],
                "Null mean": r["null_mean"],
                "Null std": r["null_std"],
                "z-score": r["z_score"],
                "p-value (one-sided)": r["p_one_sided"],
                "B": r["n_valid"],
            })
    df = pd.DataFrame(rows).sort_values(["Model", "Fold"])
    out_csv = ROOT / "paper" / "tables" / "T_permutation.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    return out_csv

def main():
    t0 = time.time()
    print("=" * 78)
    print("E4 — DeLong + block bootstrap + label permutation on test set")
    print(f"seed={SEED}, folds={FOLDS}, lookback={LOOKBACK}")
    print("=" * 78)

    print("\n[1/4] Re-train best configs and collect test predictions ...")
    preds = train_all_and_predict()

    print("\n[2/4] DeLong tests per fold (NOLT vs each baseline) ...")
    delong_out = run_delong_per_fold(preds)
    for k, v in delong_out.items():
        print(f"  Fold {k}:")
        for b, r in v.items():
            print(f"    NOLT vs {b:<22} z={r['z']:+.3f} p_1s={r['p_one_sided']:.4f} "
                  f"p_Holm={r.get('p_holm', float('nan')):.4f}")

    print("\n[3/4] Stationary block bootstrap CI (B=1000, block_len=7) ...")
    boot_out = run_bootstrap_per_fold(preds, B=1000)

    print("\n[4/4] Label permutation test (B=10000) ...")
    perm_out = run_permutation_per_fold(preds, B=10000)

    out_path = ROOT / "results" / "E4_test_stats.json"
    payload = {"seed": SEED, "folds": FOLDS, "lookback": LOOKBACK,
                "predictions": preds,
                "delong": delong_out,
                "bootstrap": boot_out,
                "permutation": perm_out}
    out_path.write_text(json.dumps(payload, indent=2,
                                     default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_path}")

    csv1 = save_delong_csv(delong_out)
    csv2 = save_bootstrap_csv(boot_out)
    csv3 = save_permutation_csv(perm_out)
    print(f"saved: {csv1}")
    print(f"saved: {csv2}")
    print(f"saved: {csv3}")
    print(f"\nelapsed: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
