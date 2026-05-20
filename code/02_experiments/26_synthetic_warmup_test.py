from __future__ import annotations

import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

from src.data.loader_pc1 import build_pc1_bundle_for_fold
from src.models.nolt import NOLT, NOLTConfig
from src.models.nolt_ablations import NOLTNoSequence
from src.models.baselines.lstm_single import LSTMSingleOption, LSTMConfig
from src.training.trainer import set_deterministic, train as train_dl, predict_proba, TrainConfig

SEED = 2026
LOOKBACK = 60
FOLDS = [3, 4, 5]
FOLD_INDICES = {3: (147, 167, 207), 4: (187, 207, 247), 5: (227, 247, 287)}
THRESHOLD_QUANTILE = 0.85
HESTON_NPZ = ROOT / "data" / "synthetic" / "heston" / "heston_panel.npz"

ARCH = {
    "NOLT":             {"d_model": 32,  "n_layers": 3, "dropout": 0.3, "lr": 3e-4},
    "NOLT_w_lookback":  {"d_model": 128, "n_layers": 2, "dropout": 0.2, "lr": 3e-4},
    "LSTM":             {"hidden_dim": 64, "n_layers": 2, "dropout": 0.1, "lr": 1e-4},
}

class SynthDS(Dataset):
    def __init__(self, R, dpc1, path_idx, threshold, lookback):
        self.R = R
        self.dpc1 = dpc1
        self.threshold = float(threshold)
        self.lookback = int(lookback)
        T = R.shape[1]

        ts = np.arange(lookback - 1, T - 1)
        self._paths = np.repeat(path_idx, len(ts))
        self._ts = np.tile(ts, len(path_idx))

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, i):
        p = int(self._paths[i]); t = int(self._ts[i])
        w = self.R[p, t - self.lookback + 1: t + 1, :]
        y = float(abs(self.dpc1[p, t]) > self.threshold)
        return torch.from_numpy(w.astype(np.float32)), torch.tensor([y], dtype=torch.float32)

def build_synth_datasets():
    arr = np.load(HESTON_NPZ, allow_pickle=True)
    R = arr["R"]
    dpc1 = arr["dpc1"]
    train_paths = arr["train_paths"]
    val_paths = arr["val_paths"]
    test_paths = arr["test_paths"]

    train_dpc1 = dpc1[train_paths].flatten()
    threshold = float(np.quantile(np.abs(train_dpc1), 0.90))

    ds_tr = SynthDS(R, dpc1, train_paths, threshold, LOOKBACK)
    ds_va = SynthDS(R, dpc1, val_paths, threshold, LOOKBACK)
    ds_te = SynthDS(R, dpc1, test_paths, threshold, LOOKBACK)
    info = {"n_paths_total": int(R.shape[0]),
            "n_paths_train": int(len(train_paths)),
            "n_paths_val": int(len(val_paths)),
            "n_paths_test": int(len(test_paths)),
            "n_train": len(ds_tr), "n_val": len(ds_va), "n_test": len(ds_te),
            "threshold": threshold, "lookback": LOOKBACK}
    return ds_tr, ds_va, ds_te, info

def _auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    ys, ss = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit = model(xb).cpu().numpy().squeeze(-1)
        ss.append(logit); ys.append(yb.numpy().squeeze(-1))
    s = np.concatenate(ss); y = np.concatenate(ys)
    return _auc(y, s)

def pretrain(model, ds_tr, ds_va, ds_te, lr=5e-4, batch=256, epochs=30, patience=8):
    set_deterministic(SEED)
    device = "cpu"
    model = model.to(device)
    g = torch.Generator(); g.manual_seed(SEED)
    L_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, generator=g)
    L_va = DataLoader(ds_va, batch_size=batch, shuffle=False)
    L_te = DataLoader(ds_te, batch_size=batch, shuffle=False)

    n_sample = min(5000, len(ds_tr))
    pos_count = sum(int(ds_tr[i][1].item()) for i in range(n_sample))
    p = pos_count / n_sample
    pos_weight = torch.tensor([(1 - p) / max(p, 1e-6)], device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best_val = -np.inf; best_state = None; best_ep = -1; no_improve = 0
    tr_hist, va_hist = [], []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0; nseen = 0
        for xb, yb in L_tr:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += loss.item() * xb.size(0); nseen += xb.size(0)
        ep_loss /= max(nseen, 1)
        v = _eval(model, L_va, device)
        t_auc = _eval(model, L_tr, device)
        tr_hist.append(t_auc); va_hist.append(v)
        if v > best_val:
            best_val = v
            best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
            best_ep = ep; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    t_auc = _eval(model, L_te, device)
    return {
        "synth_val_best": best_val,
        "synth_test": t_auc,
        "best_epoch": best_ep,
        "epochs_run": len(tr_hist),
        "train_auc_history": tr_hist,
        "val_auc_history": va_hist,
        "state_dict": best_state,
    }

def make_model(name, n_options):
    if name == "NOLT":
        cfg = ARCH["NOLT"]
        return NOLTNoSequence(NOLTConfig(n_options=n_options, lookback=LOOKBACK, n_features=1,
                                          d_model=cfg["d_model"], n_heads=4,
                                          n_layers=cfg["n_layers"], dropout=cfg["dropout"]))
    if name == "NOLT_w_lookback":
        cfg = ARCH["NOLT_w_lookback"]
        return NOLT(NOLTConfig(n_options=n_options, lookback=LOOKBACK, n_features=1,
                                 d_model=cfg["d_model"], n_heads=4,
                                 n_layers=cfg["n_layers"], dropout=cfg["dropout"]))
    if name == "LSTM":
        cfg = ARCH["LSTM"]
        return LSTMSingleOption(LSTMConfig(n_options=n_options, lookback=LOOKBACK,
                                            hidden_dim=cfg["hidden_dim"],
                                            n_layers=cfg["n_layers"],
                                            dropout=cfg["dropout"]))
    raise ValueError(name)

def finetune_on_fold(state_dict, name, bundle):
    set_deterministic(SEED)
    model = make_model(name, bundle.n_options)
    if state_dict is not None:

        model.load_state_dict(state_dict, strict=False)
    cfg = ARCH[name]
    lr_ft = cfg["lr"]
    info = train_dl(model, bundle.X_train, bundle.y_train,
                     bundle.X_val, bundle.y_val,
                     TrainConfig(epochs=60, batch_size=32, lr=lr_ft,
                                  weight_decay=1e-3, early_stop_patience=20, seed=SEED))
    s = predict_proba(model, bundle.X_test)
    return {
        "test_auc": _auc(bundle.y_test, s),
        "best_val_auc": float(info["best_val_auc"]),
        "best_epoch": int(info["best_epoch"]),
        "epochs_run": int(info["epochs_run"]),
        "test_predictions": s.tolist(),
        "test_labels": bundle.y_test.tolist(),
    }

def main():
    t0 = time.time()
    print("=" * 78)
    print("E3 — Pretraining fairness ablation")
    print(f"seed={SEED}, folds={FOLDS}, lookback={LOOKBACK}")
    print(f"Models: NOLT, NOLT_w_lookback, LSTM (architectures from T1 best configs)")
    print("=" * 78)

    print("\n[Phase 1] Build Heston synth datasets ...")
    ds_tr, ds_va, ds_te, synth_info = build_synth_datasets()
    print(f"  paths: train={synth_info['n_paths_train']}, val={synth_info['n_paths_val']}, test={synth_info['n_paths_test']}")
    print(f"  samples: train={synth_info['n_train']}, val={synth_info['n_val']}, test={synth_info['n_test']}")
    print(f"  threshold={synth_info['threshold']:.6f}")

    print("\n[Phase 2] Build real fold bundles ...")
    bundles = {k: build_pc1_bundle_for_fold(*FOLD_INDICES[k], lookback=LOOKBACK,
                                               horizon=1,
                                               threshold_quantile=THRESHOLD_QUANTILE,
                                               seed=SEED) for k in FOLDS}
    for k, b in bundles.items():
        print(f"  Fold {k}: train={len(b.X_train)} val={len(b.X_val)} test={len(b.X_test)}")

    e4_path = ROOT / "results" / "E4_test_stats.json"
    e4 = json.loads(e4_path.read_text()) if e4_path.exists() else None

    print("\n[Phase 3] Pretrain each model on Heston synth ...")
    pretrain_results = {}
    for name in ["NOLT", "NOLT_w_lookback", "LSTM"]:
        t_m = time.time()
        print(f"\n  --- {name} pretrain ---")
        model = make_model(name, n_options=27)

        if name == "LSTM":
            res = pretrain(model, ds_tr, ds_va, ds_te, lr=3e-4, batch=256, epochs=12, patience=6)
        else:
            res = pretrain(model, ds_tr, ds_va, ds_te, lr=5e-4, batch=256, epochs=15, patience=6)
        print(f"    synth val best AUC={res['synth_val_best']:.4f}, test AUC={res['synth_test']:.4f}, "
              f"epochs_run={res['epochs_run']} ({time.time()-t_m:.0f}s)")
        pretrain_results[name] = res

    print("\n[Phase 4] Fine-tune each pretrained model on each real fold ...")
    finetune_results = {}
    for name in ["NOLT", "NOLT_w_lookback", "LSTM"]:
        finetune_results[name] = {}
        for k, b in bundles.items():
            t_f = time.time()
            r = finetune_on_fold(pretrain_results[name]["state_dict"], name, b)
            print(f"  {name} fold {k}: test_AUC={r['test_auc']:.4f}, val_best={r['best_val_auc']:.4f}, "
                  f"epochs={r['epochs_run']} ({time.time()-t_f:.0f}s)")
            finetune_results[name][str(k)] = r

    print("\n=== SUMMARY (with vs without pretrain) ===")
    rows = []
    for name in ["NOLT", "NOLT_w_lookback", "LSTM"]:
        without_aucs = {}
        if e4 and name in e4["predictions"]:
            for k_str, d in e4["predictions"][name].items():
                without_aucs[int(k_str)] = float(d["auc"]) if d["auc"] is not None else float("nan")
        with_aucs = {int(k): float(finetune_results[name][k]["test_auc"]) for k in finetune_results[name]}
        for k in FOLDS:
            rows.append({
                "Model": name,
                "Fold": k,
                "Without pretrain AUC": without_aucs.get(k, float("nan")),
                "With pretrain AUC": with_aucs.get(k, float("nan")),
                "Delta (with - without)": with_aucs.get(k, float("nan")) - without_aucs.get(k, float("nan")),
            })
        med_with = float(np.median([with_aucs[k] for k in FOLDS]))
        med_without = float(np.median([without_aucs.get(k, float("nan")) for k in FOLDS]))
        rows.append({
            "Model": name + " (median)",
            "Fold": "median",
            "Without pretrain AUC": med_without,
            "With pretrain AUC": med_with,
            "Delta (with - without)": med_with - med_without,
        })
        print(f"  {name}: median without={med_without:.4f}, with={med_with:.4f}, "
              f"delta={med_with-med_without:+.4f}")

    df = pd.DataFrame(rows)
    out_csv = ROOT / "paper" / "tables" / "T_pretrain_fairness.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nsaved: {out_csv}")

    pt_summary = {k: {kk: vv for kk, vv in v.items() if kk != "state_dict"}
                   for k, v in pretrain_results.items()}
    out_json = ROOT / "results" / "E3_pretrain_fairness.json"
    out_json.write_text(json.dumps({
        "seed": SEED, "folds": FOLDS, "lookback": LOOKBACK,
        "synth_info": synth_info,
        "pretrain_results": pt_summary,
        "finetune_results": finetune_results,
        "summary_rows": rows,
    }, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"saved: {out_json}")
    print(f"\nelapsed: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
