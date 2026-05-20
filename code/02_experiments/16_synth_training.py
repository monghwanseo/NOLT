from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HESTON_NPZ = ROOT / "data" / "synthetic" / "heston" / "heston_panel.npz"
BATES_NPZ  = ROOT / "data" / "synthetic" / "bates"  / "bates_panel.npz"

SEED = 2026
LOOKBACK = 20
DEVICE = "cpu"
BATCH = 256

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auroc(y_true, scores):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))

def build_classification_samples(panel_npz_path, lookback=LOOKBACK):
    data = np.load(panel_npz_path, allow_pickle=True)
    R = np.asarray(data["R"])
    sigma_iv = np.asarray(data["sigma_iv"])
    mny = np.asarray(data["moneyness"])
    tau = np.asarray(data["tau"])
    pc1 = np.asarray(data["pc1"])
    labels = np.asarray(data["labels"])

    P, T, N = R.shape
    train_paths = np.asarray(data["train_paths"])
    val_paths = np.asarray(data["val_paths"])
    test_paths = np.asarray(data["test_paths"])
    threshold = float(np.asarray(data["threshold"]))
    K_arr = np.asarray(data["K"])
    type_arr = np.asarray(data["option_type"])

    atm_idx = int(np.argmin(np.abs(mny[0, 0])))

    samples = []
    valid_T = T - 1
    for p in range(P):
        for t in range(lookback, valid_T):
            samples.append((p, t))
    samples = np.asarray(samples, dtype=np.int32)

    P_idx = samples[:, 0]; T_idx = samples[:, 1]
    y = labels[P_idx, T_idx].astype(np.float32)

    X_nolt = R[P_idx, T_idx, :].astype(np.float32)

    X_nolt = np.stack([X_nolt, sigma_iv[P_idx, T_idx, :].astype(np.float32),
                       mny[P_idx, T_idx, :].astype(np.float32)], axis=-1)

    X_xgb = np.concatenate([R[P_idx, T_idx, :], sigma_iv[P_idx, T_idx, :],
                             mny[P_idx, T_idx, :]], axis=1).astype(np.float32)

    X_lstm = np.empty((len(samples), lookback, 3), dtype=np.float32)
    for k, (p, t) in enumerate(samples):
        X_lstm[k, :, 0] = R[p, t - lookback:t, atm_idx]
        X_lstm[k, :, 1] = sigma_iv[p, t - lookback:t, atm_idx]
        X_lstm[k, :, 2] = mny[p, t - lookback:t, atm_idx]

    PC1_hist = np.empty((len(samples), lookback), dtype=np.float32)
    for k, (p, t) in enumerate(samples):
        PC1_hist[k, :] = pc1[p, t - lookback:t]

    train_mask = np.isin(P_idx, train_paths)
    val_mask = np.isin(P_idx, val_paths)
    test_mask = np.isin(P_idx, test_paths)

    return {
        "X_lstm": X_lstm, "X_nolt": X_nolt, "X_xgb": X_xgb,
        "PC1_hist": PC1_hist, "y": y,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "threshold": threshold, "atm_idx": atm_idx, "N_opt": N,
    }

class LSTMSingleModel(nn.Module):
    def __init__(self, in_dim=3, hidden=64, n_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=n_layers,
                              dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :]).squeeze(-1)

class NOLTSnap(nn.Module):
    def __init__(self, n_opt=27, in_feat=3, d_model=32, n_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_feat, d_model)

        self.pos = nn.Parameter(torch.randn(1, n_opt, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                  dim_feedforward=2 * d_model, dropout=dropout,
                                                  batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.proj(x) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)

def train_torch_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       lr=3e-4, batch=BATCH, max_epochs=40, patience=10, weight_decay=1e-3,
                       device=DEVICE, log_prefix=""):
    set_seed(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_rate = float(y_train.mean())
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.from_numpy(X_train).to(device); yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device); yv = torch.from_numpy(y_val).to(device)
    Xs = torch.from_numpy(X_test).to(device); ys = torch.from_numpy(y_test).to(device)

    best_val = -np.inf; best_test = np.nan; best_epoch = -1; bad = 0
    n_train = len(Xt)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = crit(logits, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(Xv).cpu().numpy()
            t_logits = model(Xs).cpu().numpy()
        v_auc = auroc(y_val, v_logits)
        t_auc = auroc(y_test, t_logits)
        if v_auc > best_val:
            best_val = v_auc; best_test = t_auc; best_epoch = epoch; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return {"val_auc": best_val, "test_auc": best_test, "best_epoch": best_epoch + 1,
            "epochs_run": epoch + 1}

def fit_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, **kw):
    from xgboost import XGBClassifier
    set_seed(SEED)
    model = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="auc",
                          use_label_encoder=False, **kw)
    model.fit(X_train, y_train.astype(int))
    v_score = model.predict_proba(X_val)[:, 1]
    t_score = model.predict_proba(X_test)[:, 1]
    return {"val_auc": auroc(y_val, v_score), "test_auc": auroc(y_test, t_score)}

def bsm_threshold_rule(PC1_train, y_train, PC1_val, y_val, PC1_test, y_test, tail_window=15):
    set_seed(SEED)
    def feature(PC1_hist):

        diffs = np.abs(np.diff(PC1_hist, axis=1))
        return diffs[:, -tail_window:].max(axis=1)
    f_train = feature(PC1_train); f_val = feature(PC1_val); f_test = feature(PC1_test)
    return {"val_auc": auroc(y_val, f_val), "test_auc": auroc(y_test, f_test)}

def sweep_lstm(d, log_prefix=""):
    Xtr = d["X_lstm"][d["train_mask"]]; ytr = d["y"][d["train_mask"]]
    Xva = d["X_lstm"][d["val_mask"]]; yva = d["y"][d["val_mask"]]
    Xte = d["X_lstm"][d["test_mask"]]; yte = d["y"][d["test_mask"]]
    grid = []
    for h in [32, 64]:
        for L in [1, 2]:
            for dr in [0.1, 0.2]:
                for lr in [3e-4, 1e-3]:
                    grid.append({"hidden": h, "n_layers": L, "dropout": dr, "lr": lr})
    print(f"  [LSTM] sweep grid size = {len(grid)}", flush=True)
    results = {}; best = None; best_score = -np.inf
    for g in grid:
        t0 = time.time()
        model = LSTMSingleModel(in_dim=3, hidden=g["hidden"], n_layers=g["n_layers"], dropout=g["dropout"])
        r = train_torch_model(model, Xtr, ytr, Xva, yva, Xte, yte,
                                lr=g["lr"], batch=BATCH, max_epochs=30, patience=8,
                                log_prefix=log_prefix + f"[LSTM h={g['hidden']} L={g['n_layers']}]")
        cfg_str = f"h={g['hidden']},L={g['n_layers']},dr={g['dropout']},lr={g['lr']}"
        results[cfg_str] = {**r, **g}
        print(f"    [LSTM {cfg_str}] val={r['val_auc']:.4f} test={r['test_auc']:.4f} "
              f"epoch={r['best_epoch']} ({time.time()-t0:.0f}s)", flush=True)
        if r["val_auc"] > best_score:
            best_score = r["val_auc"]; best = (cfg_str, r)
    return {"all": results, "best": {"config": best[0], **best[1]}}

def sweep_nolt(d, log_prefix=""):
    Xtr = d["X_nolt"][d["train_mask"]]; ytr = d["y"][d["train_mask"]]
    Xva = d["X_nolt"][d["val_mask"]]; yva = d["y"][d["val_mask"]]
    Xte = d["X_nolt"][d["test_mask"]]; yte = d["y"][d["test_mask"]]
    grid = []
    for d_m in [32, 64]:
        for L in [2, 3]:
            for dr in [0.1, 0.2, 0.3]:
                for lr in [3e-4, 1e-3]:
                    grid.append({"d_model": d_m, "n_layers": L, "dropout": dr, "lr": lr})
    print(f"  [NOLT] sweep grid size = {len(grid)}", flush=True)
    results = {}; best = None; best_score = -np.inf
    for g in grid:
        t0 = time.time()
        model = NOLTSnap(n_opt=d["N_opt"], in_feat=3, d_model=g["d_model"],
                          n_layers=g["n_layers"], n_heads=4, dropout=g["dropout"])
        r = train_torch_model(model, Xtr, ytr, Xva, yva, Xte, yte,
                                lr=g["lr"], batch=BATCH, max_epochs=30, patience=8)
        cfg_str = f"d={g['d_model']},L={g['n_layers']},dr={g['dropout']},lr={g['lr']}"
        results[cfg_str] = {**r, **g}
        print(f"    [NOLT {cfg_str}] val={r['val_auc']:.4f} test={r['test_auc']:.4f} "
              f"epoch={r['best_epoch']} ({time.time()-t0:.0f}s)", flush=True)
        if r["val_auc"] > best_score:
            best_score = r["val_auc"]; best = (cfg_str, r)
    return {"all": results, "best": {"config": best[0], **best[1]}}

def sweep_xgb(d):
    Xtr = d["X_xgb"][d["train_mask"]]; ytr = d["y"][d["train_mask"]]
    Xva = d["X_xgb"][d["val_mask"]]; yva = d["y"][d["val_mask"]]
    Xte = d["X_xgb"][d["test_mask"]]; yte = d["y"][d["test_mask"]]
    grid = [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03},
    ]
    results = {}; best = None; best_score = -np.inf
    for g in grid:
        t0 = time.time()
        r = fit_xgboost(Xtr, ytr, Xva, yva, Xte, yte, **g)
        cfg_str = f"n={g['n_estimators']},d={g['max_depth']},lr={g['learning_rate']}"
        results[cfg_str] = {**r, **g}
        print(f"    [XGB {cfg_str}] val={r['val_auc']:.4f} test={r['test_auc']:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        if r["val_auc"] > best_score:
            best_score = r["val_auc"]; best = (cfg_str, r)
    return {"all": results, "best": {"config": best[0], **best[1]}}

def sweep_bsm(d):
    PCtr = d["PC1_hist"][d["train_mask"]]; ytr = d["y"][d["train_mask"]]
    PCva = d["PC1_hist"][d["val_mask"]]; yva = d["y"][d["val_mask"]]
    PCte = d["PC1_hist"][d["test_mask"]]; yte = d["y"][d["test_mask"]]
    grid = [3, 5, 10, 15]
    results = {}; best = None; best_score = -np.inf
    for tw in grid:
        r = bsm_threshold_rule(PCtr, ytr, PCva, yva, PCte, yte, tail_window=tw)
        cfg_str = f"tw={tw}"
        results[cfg_str] = {**r, "tail_window": tw}
        print(f"    [BSM tw={tw}] val={r['val_auc']:.4f} test={r['test_auc']:.4f}", flush=True)
        if r["val_auc"] > best_score:
            best_score = r["val_auc"]; best = (cfg_str, r)
    return {"all": results, "best": {"config": best[0], **best[1]}}

def run_dataset(name, npz_path):
    print(f"\n{'='*70}\n[B8c] Training on {name} ({npz_path})\n{'='*70}", flush=True)
    d = build_classification_samples(npz_path)
    print(f"  Samples: train={d['train_mask'].sum()}, val={d['val_mask'].sum()}, test={d['test_mask'].sum()}",
          flush=True)
    print(f"  Pos rate (train/val/test): {d['y'][d['train_mask']].mean():.3f} / "
          f"{d['y'][d['val_mask']].mean():.3f} / {d['y'][d['test_mask']].mean():.3f}", flush=True)

    out = {}
    print("\n  --- BSM threshold ---", flush=True)
    out["bsm_threshold"] = sweep_bsm(d)
    print("\n  --- XGBoost ---", flush=True)
    out["xgboost"] = sweep_xgb(d)
    print("\n  --- LSTM-single ---", flush=True)
    out["lstm_single"] = sweep_lstm(d)
    print("\n  --- NOLT-snap ---", flush=True)
    out["nolt_snap"] = sweep_nolt(d)
    return out

def main():
    set_seed(SEED)
    print(f"[B8c] seed={SEED}, lookback={LOOKBACK}, batch={BATCH}", flush=True)

    out = {}
    out["heston"] = run_dataset("HESTON (single-latent)", HESTON_NPZ)
    out["bates"]  = run_dataset("BATES  (multi-latent)",   BATES_NPZ)

    out["seed"] = SEED
    out["lookback"] = LOOKBACK

    print("\n" + "="*70 + "\n[B8c] SUMMARY (best val AUROC per model)\n" + "="*70, flush=True)
    print(f"{'Model':<18} {'Heston val':>12} {'Heston test':>12} {'Bates val':>12} {'Bates test':>12}",
          flush=True)
    for m in ["bsm_threshold", "xgboost", "lstm_single", "nolt_snap"]:
        h = out["heston"][m]["best"]; b = out["bates"][m]["best"]
        print(f"{m:<18} {h['val_auc']:>12.4f} {h['test_auc']:>12.4f} "
              f"{b['val_auc']:>12.4f} {b['test_auc']:>12.4f}", flush=True)

    OUT = RESULTS / "synth_training.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[B8c] Wrote {OUT}", flush=True)

if __name__ == "__main__":
    main()
