from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.models.nolt import NOLT, NOLTConfig

SEED = 2026
DEVICE = "cpu"
BATCH = 256
LOOKBACK_FULL = 60
ARCH = {"d_model": 32, "n_layers": 3, "dropout": 0.2}
HESTON_NPZ = ROOT / "data" / "synthetic" / "heston" / "heston_panel.npz"
BATES_NPZ = ROOT / "data" / "synthetic" / "bates" / "bates_panel.npz"

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def build_temporal_samples(npz_path: Path, lookback: int):
    data = np.load(npz_path, allow_pickle=True)
    R = np.asarray(data["R"])
    sigma_iv = np.asarray(data["sigma_iv"])
    mny = np.asarray(data["moneyness"])
    labels = np.asarray(data["labels"])
    train_paths = np.asarray(data["train_paths"])
    val_paths = np.asarray(data["val_paths"])
    test_paths = np.asarray(data["test_paths"])

    P, T, N = R.shape
    valid_T = T - 1

    samples = []
    for p in range(P):
        for t in range(lookback, valid_T):
            samples.append((p, t))
    samples = np.asarray(samples, dtype=np.int32)
    P_idx = samples[:, 0]; T_idx = samples[:, 1]

    y = labels[P_idx, T_idx].astype(np.float32)

    X = np.empty((len(samples), lookback, N, 3), dtype=np.float32)
    for k, (p, t) in enumerate(samples):
        X[k, :, :, 0] = R[p, t-lookback:t, :]
        X[k, :, :, 1] = sigma_iv[p, t-lookback:t, :]
        X[k, :, :, 2] = mny[p, t-lookback:t, :]

    train_mask = np.isin(P_idx, train_paths)
    val_mask = np.isin(P_idx, val_paths)
    test_mask = np.isin(P_idx, test_paths)

    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_val": X[val_mask], "y_val": y[val_mask],
        "X_test": X[test_mask], "y_test": y[test_mask],
        "N_opt": int(N), "lookback": lookback,
    }

def train_nolt_full(samples, lookback, arch, lr=3e-4, max_epochs=40, patience=10,
                     weight_decay=1e-3, log_prefix=""):
    set_seed(SEED)
    n_opt = samples["N_opt"]
    cfg = NOLTConfig(n_options=n_opt, lookback=lookback, n_features=3,
                     d_model=arch["d_model"], n_heads=4, n_layers=arch["n_layers"],
                     dropout=arch["dropout"])
    model = NOLT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_rate = float(samples["y_train"].mean())
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.from_numpy(samples["X_train"]).to(DEVICE)
    yt = torch.from_numpy(samples["y_train"]).to(DEVICE)
    Xv = torch.from_numpy(samples["X_val"]).to(DEVICE)
    yv = torch.from_numpy(samples["y_val"]).to(DEVICE)
    Xs = torch.from_numpy(samples["X_test"]).to(DEVICE)
    ys = torch.from_numpy(samples["y_test"]).to(DEVICE)

    best_val = -np.inf; best_test = float("nan"); best_epoch = -1; bad = 0
    n_train = len(Xt)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            logits = model(Xt[idx]).squeeze(-1)
            loss = crit(logits, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = []
            for i in range(0, len(Xv), BATCH):
                sv.append(model(Xv[i:i+BATCH]).squeeze(-1).cpu().numpy())
            sv = np.concatenate(sv)
            ss = []
            for i in range(0, len(Xs), BATCH):
                ss.append(model(Xs[i:i+BATCH]).squeeze(-1).cpu().numpy())
            ss = np.concatenate(ss)
        val_auc = auroc(samples["y_val"], sv)
        test_auc = auroc(samples["y_test"], ss)
        if val_auc > best_val:
            best_val = val_auc; best_test = test_auc; best_epoch = epoch; bad = 0
        else:
            bad += 1
        if epoch < 3 or epoch % 5 == 0 or bad == 0:
            print(f"  {log_prefix}epoch {epoch:3d}: val={val_auc:.4f} test={test_auc:.4f} "
                  f"best_val={best_val:.4f}")
        if bad >= patience: break
    return {"val_auc": float(best_val), "test_auc": float(best_test),
            "best_epoch": int(best_epoch), "epochs_run": int(epoch+1)}

def main():
    t0 = time.time()
    print("=" * 70)
    print(f"Phase 2 #7b - NOLT-full on synth (d=32, L=3, dr=0.2, lookback={LOOKBACK_FULL})")
    print(f"Seed=2026, within-synth uniform config (Heston == Bates)")
    print("=" * 70)

    out = {"arch": ARCH, "lookback": LOOKBACK_FULL, "seed": SEED, "results": {}}
    for name, npz in [("heston", HESTON_NPZ), ("bates", BATES_NPZ)]:
        print(f"\n[{name.upper()}] building samples (lookback={LOOKBACK_FULL}) ...")
        samples = build_temporal_samples(npz, LOOKBACK_FULL)
        print(f"  X_train shape: {samples['X_train'].shape}")
        print(f"  X_val   shape: {samples['X_val'].shape}")
        print(f"  X_test  shape: {samples['X_test'].shape}")
        print(f"  Training NOLT-full ...")
        res = train_nolt_full(samples, LOOKBACK_FULL, ARCH, log_prefix=f"[{name}] ")
        out["results"][name] = res
        print(f"  {name}: val={res['val_auc']:.4f}, test={res['test_auc']:.4f}")

    out["elapsed_seconds"] = time.time() - t0
    out_path = ROOT / "results" / "phase2_synth_nolt_full.json"
    out_path.write_text(json.dumps(out, indent=2,
                                      default=lambda x: float(x) if isinstance(x, np.floating) else x))
    print(f"\nsaved: {out_path}  ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
