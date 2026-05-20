import json, sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.data.loader_pc1 import build_pc1_bundle_for_fold
from src.models.nolt_ablations import NOLTNoSequence
from src.models.nolt import NOLTConfig
from src.training.trainer import set_deterministic
from _consistency_locks import REAL_BEST, SEED, LOOKBACK, THRESHOLD_QUANTILE_REAL

OUT = ROOT / "results"
FOLDS = [3, 4, 5]
FOLD_INDICES = {3: (147, 167, 207), 4: (187, 207, 247), 5: (227, 247, 287)}

def auc(y, s):
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, s))

def train_with_curves(model, X_tr, y_tr, X_v, y_v, X_te, y_te, lr, batch, epochs, patience, wd):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    p_pos = float(y_tr.mean())
    pos_w = torch.tensor([(1 - p_pos)/max(p_pos, 1e-6)], device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(X_tr).float().to(device); yt = torch.from_numpy(y_tr).float().unsqueeze(-1).to(device)
    Xv = torch.from_numpy(X_v).float().to(device); yv_np = y_v
    Xe = torch.from_numpy(X_te).float().to(device)
    g = torch.Generator(); g.manual_seed(SEED)
    n = len(Xt)
    train_curve, val_curve, test_curve = [], [], []
    best_val = -np.inf; best_state = None; best_ep = -1; no_improve = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xt[idx], yt[idx]
            optim.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval()
        with torch.no_grad():
            tr_logits = model(Xt).cpu().numpy().squeeze(-1)
            v_logits = model(Xv).cpu().numpy().squeeze(-1)
            te_logits = model(Xe).cpu().numpy().squeeze(-1)
        ta = auc(y_tr, tr_logits); va = auc(y_v, v_logits); tea = auc(y_te, te_logits)
        train_curve.append(ta); val_curve.append(va); test_curve.append(tea)
        if not np.isnan(va) and va > best_val:
            best_val = va; best_ep = ep; no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = test_curve[best_ep] if best_ep >= 0 else float("nan")
    return {"train_curve": train_curve, "val_curve": val_curve, "test_curve": test_curve,
            "best_epoch": best_ep, "best_val": float(best_val), "test_at_best_val": float(final_test),
            "epochs_run": len(train_curve), "max_test": float(max(test_curve)) if test_curve else float("nan")}

def main():
    print("[P1.4] NOLT-snap convergence sanity (Window A, locked config)", flush=True)
    cfg = REAL_BEST["nolt_snap"]
    out = {}
    for k in FOLDS:
        b = build_pc1_bundle_for_fold(*FOLD_INDICES[k], lookback=LOOKBACK, horizon=1,
                                        threshold_quantile=THRESHOLD_QUANTILE_REAL, seed=SEED)
        set_deterministic(SEED)
        m = NOLTNoSequence(NOLTConfig(n_options=b.n_options, lookback=LOOKBACK, n_features=1,
                                        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                                        n_layers=cfg["n_layers"], dropout=cfg["dropout"]))
        r = train_with_curves(m, b.X_train, b.y_train, b.X_val, b.y_val, b.X_test, b.y_test,
                                cfg["lr"], cfg["batch"], cfg["max_epochs"], cfg["patience"], cfg["weight_decay"])
        out[k] = r
        print(f"  Fold {k}: best_ep={r['best_epoch']}/{r['epochs_run']} "
              f"train@best={r['train_curve'][r['best_epoch']]:.4f}  val={r['best_val']:.4f}  "
              f"test={r['test_at_best_val']:.4f}  max_test={r['max_test']:.4f}  "
              f"overfit_gap={r['train_curve'][r['best_epoch']]-r['test_at_best_val']:+.4f}", flush=True)

    test_at_best = [out[k]["test_at_best_val"] for k in FOLDS]
    out["median_test"] = float(np.median(test_at_best))
    print(f"\n  median test = {out['median_test']:.4f} across folds {test_at_best}", flush=True)
    if out["median_test"] >= 0.70:
        print("  -> NOLT-snap robustly above 0.70 on all valid folds", flush=True)
    with open(OUT / "p1_nolt_snap_sanity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved: p1_nolt_snap_sanity.json", flush=True)

if __name__ == "__main__":
    main()
