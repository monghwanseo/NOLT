from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
COLOR_REAL = "#000000"
COLOR_HESTON = "#0072B2"
COLOR_BATES = "#D55E00"
COLOR_NULL = "#999999"

def load_json(p):
    with open(p) as f:
        return json.load(f)

def random_null_var_share(N=27, T=349, n_trials=200, seed=SEED):
    rng = np.random.default_rng(seed)
    shares_pc1, shares_pc2, shares_pc3 = [], [], []
    for _ in range(n_trials):
        R = rng.standard_normal((T, N))
        Rc = R - R.mean(axis=0, keepdims=True)
        cov = Rc.T @ Rc / max(T - 1, 1)
        ev = np.linalg.eigvalsh(cov)[::-1]
        total = ev.sum()
        shares_pc1.append(ev[0] / total)
        shares_pc2.append(ev[1] / total)
        shares_pc3.append(ev[2] / total)
    return float(np.median(shares_pc1)), float(np.median(shares_pc2)), float(np.median(shares_pc3))

def main():
    h = load_json(RESULTS / "heston_iv_matched.json")["configs"]
    b = load_json(RESULTS / "bates_phenomenon.json")["configs"]
    m1 = load_json(RESULTS / "M1_summary.json")

    real = {"pc1": m1["M6"]["pc1_var"], "pc1_acf": m1["M6"]["pc1_acf1"]}
    h_pc = {
        "pc1": float(np.median([c["var_share_median"] for c in h.values()])),
        "pc1_acf": float(np.median([c["acf1_median"] for c in h.values()])),
        "pc2": float(np.median([c["var_share_pc2_median"] for c in h.values()])),
        "pc2_acf": float(np.median([c["acf1_pc2_median"] for c in h.values()])),
        "pc3": float(np.median([c["var_share_pc3_median"] for c in h.values()])),
        "pc3_acf": float(np.median([c["acf1_pc3_median"] for c in h.values()])),
    }
    b_pc = {
        "pc1": float(np.median([c["var_share_pc1_median"] for c in b.values()])),
        "pc1_acf": float(np.median([c["acf1_pc1_median"] for c in b.values()])),
        "pc2": float(np.median([c["var_share_pc2_median"] for c in b.values()])),
        "pc2_acf": float(np.median([c["acf1_pc2_median"] for c in b.values()])),
        "pc3": float(np.median([c["var_share_pc3_median"] for c in b.values()])),
        "pc3_acf": float(np.median([c["acf1_pc3_median"] for c in b.values()])),
    }
    null_p1, null_p2, null_p3 = random_null_var_share()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pcs = ["PC1", "PC2", "PC3"]
    null_vals = [null_p1, null_p2, null_p3]
    real_vals = [real["pc1"], None, None]
    h_vals = [h_pc["pc1"], h_pc["pc2"], h_pc["pc3"]]
    b_vals = [b_pc["pc1"], b_pc["pc2"], b_pc["pc3"]]
    x = np.arange(len(pcs))
    w = 0.2
    ax.bar(x - 1.5 * w, null_vals, w, color=COLOR_NULL, label="Null (Gaussian)", edgecolor="black")
    real_drawn = [v if v is not None else 0 for v in real_vals]
    bars_real = ax.bar(x - 0.5 * w, real_drawn, w, color=COLOR_REAL, label="Real (Window A)", edgecolor="black")
    ax.bar(x + 0.5 * w, h_vals, w, color=COLOR_HESTON, label="Heston (single-latent)", edgecolor="black")
    ax.bar(x + 1.5 * w, b_vals, w, color=COLOR_BATES, label="Bates (multi-latent)", edgecolor="black")

    for i, (n, r, hh, bb) in enumerate(zip(null_vals, real_drawn, h_vals, b_vals)):
        ax.text(i - 1.5 * w, n + 0.01, f"{n:.2f}", ha="center", fontsize=8)
        if real_vals[i] is not None:
            ax.text(i - 0.5 * w, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        else:
            ax.text(i - 0.5 * w, 0.02, "—", ha="center", fontsize=8)
        ax.text(i + 0.5 * w, hh + 0.01, f"{hh:.2f}", ha="center", fontsize=8)
        ax.text(i + 1.5 * w, bb + 0.01, f"{bb:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(pcs)
    ax.set_ylabel("Var(PC_k) / Var(total)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="grey", ls=":", lw=0.7)
    ax.set_title("PC variance share — Real vs Heston (single-latent) vs Bates (multi-latent)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "F_phenom_pc_var.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    real_acf = [real["pc1_acf"], None, None]
    h_acf = [h_pc["pc1_acf"], h_pc["pc2_acf"], h_pc["pc3_acf"]]
    b_acf = [b_pc["pc1_acf"], b_pc["pc2_acf"], b_pc["pc3_acf"]]
    real_drawn_acf = [v if v is not None else 0 for v in real_acf]
    ax.bar(x - 0.5 * w, real_drawn_acf, w, color=COLOR_REAL, label="Real (Window A)", edgecolor="black")
    ax.bar(x + 0.5 * w, h_acf, w, color=COLOR_HESTON, label="Heston", edgecolor="black")
    ax.bar(x + 1.5 * w, b_acf, w, color=COLOR_BATES, label="Bates", edgecolor="black")
    for i in range(3):
        if real_acf[i] is not None:
            ax.text(i - 0.5 * w, real_drawn_acf[i] + 0.01, f"{real_drawn_acf[i]:.2f}", ha="center", fontsize=8)
        else:
            ax.text(i - 0.5 * w, 0.02, "—", ha="center", fontsize=8)
        ax.text(i + 0.5 * w, h_acf[i] + 0.01, f"{h_acf[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + 1.5 * w, b_acf[i] + 0.01, f"{b_acf[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(pcs)
    ax.set_ylabel("ACF(1)")
    ax.set_ylim(0, 1.0)
    ax.set_title("PC autocorrelation — All persistent (PC1, PC2, PC3 all I(1)-like)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "F_phenom_acf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")

    h_adf = np.array([c["adf_p_median"] for c in h.values()])
    h_kpss = np.array([c["kpss_p_median"] for c in h.values()])
    h_acf1 = np.array([c["acf1_median"] for c in h.values()])
    h_var1 = np.array([c["var_share_median"] for c in h.values()])
    b_adf = np.array([c["adf_pc1_median"] for c in b.values()])
    b_kpss = np.array([c["kpss_pc1_median"] for c in b.values()])
    b_acf1 = np.array([c["acf1_pc1_median"] for c in b.values()])
    b_var1 = np.array([c["var_share_pc1_median"] for c in b.values()])

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    real_adf = 0.836; real_kpss = 0.010; real_acf1 = 0.968; real_var1 = 0.781

    ax = axes[0]
    ax.bar(["Real", "Heston", "Bates"], [real_adf, np.median(h_adf), np.median(b_adf)],
           color=[COLOR_REAL, COLOR_HESTON, COLOR_BATES], edgecolor="black")
    ax.axhline(0.05, color="grey", ls=":", lw=0.7)
    ax.set_ylim(0, 1.0); ax.set_ylabel("ADF p-value")
    ax.set_title("(a) ADF p-value [> 0.05 = unit root]")
    for i, v in enumerate([real_adf, np.median(h_adf), np.median(b_adf)]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    ax = axes[1]
    ax.bar(["Real", "Heston", "Bates"], [real_kpss, np.median(h_kpss), np.median(b_kpss)],
           color=[COLOR_REAL, COLOR_HESTON, COLOR_BATES], edgecolor="black")
    ax.axhline(0.05, color="grey", ls=":", lw=0.7)
    ax.set_ylim(0, 0.10); ax.set_ylabel("KPSS p-value")
    ax.set_title("(b) KPSS p-value [< 0.05 = non-stationary]")
    for i, v in enumerate([real_kpss, np.median(h_kpss), np.median(b_kpss)]):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    ax = axes[2]
    ax.bar(["Real", "Heston", "Bates"], [real_acf1, np.median(h_acf1), np.median(b_acf1)],
           color=[COLOR_REAL, COLOR_HESTON, COLOR_BATES], edgecolor="black")
    ax.set_ylim(0, 1.0); ax.set_ylabel("PC1 ACF(1)")
    ax.set_title("(c) PC1 first-order autocorrelation")
    for i, v in enumerate([real_acf1, np.median(h_acf1), np.median(b_acf1)]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    ax = axes[3]
    ax.bar(["Null", "Real", "Heston", "Bates"],
           [null_p1, real_var1, np.median(h_var1), np.median(b_var1)],
           color=[COLOR_NULL, COLOR_REAL, COLOR_HESTON, COLOR_BATES], edgecolor="black")
    ax.axhline(0.5, color="grey", ls=":", lw=0.7)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Var(PC1) / Var(total)")
    ax.set_title("(d) Cross-section variance share")
    for i, v in enumerate([null_p1, real_var1, np.median(h_var1), np.median(b_var1)]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    fig.tight_layout()
    out = FIG_DIR / "F_phenom_stationarity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")

    train_path = RESULTS / "synth_training.json"
    if train_path.exists():
        train = load_json(train_path)
        real_train = load_json(RESULTS / "real_domain_results.json")["summary"]
        abl = load_json(RESULTS / "ablation_results.json")["summary"]

        models = ["bsm_threshold", "xgboost", "lstm_single", "nolt_snap"]
        labels = ["BSM", "XGBoost", "LSTM-single", "NOLT-snap"]
        h_test = [train["heston"][m]["best"]["test_auc"] for m in models]
        b_test = [train["bates"][m]["best"]["test_auc"] for m in models]
        real_map = {"bsm_threshold": real_train["bsm_threshold"]["median_test"],
                    "xgboost": real_train["xgboost"]["median_test"],
                    "lstm_single": real_train["lstm_single"]["median_test"],
                    "nolt_snap": abl["nolt_no_sequence"]["median_test"]}
        r_test = [real_map[m] for m in models]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(labels))
        w = 0.27
        ax.bar(x - w, h_test, w, color=COLOR_HESTON, label="Heston (single-latent)", edgecolor="black")
        ax.bar(x, b_test, w, color=COLOR_BATES, label="Bates (multi-latent)", edgecolor="black")
        ax.bar(x + w, r_test, w, color=COLOR_REAL, label="Real Window A", edgecolor="black")
        for i, (h_v, b_v, r_v) in enumerate(zip(h_test, b_test, r_test)):
            ax.text(i - w, h_v + 0.01, f"{h_v:.2f}", ha="center", fontsize=8)
            ax.text(i, b_v + 0.01, f"{b_v:.2f}", ha="center", fontsize=8)
            ax.text(i + w, r_v + 0.01, f"{r_v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Test AUROC")
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="grey", ls=":", lw=0.7)
        ax.set_title("Training comparison — Heston vs Bates vs Real")
        ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout()
        out = FIG_DIR / "F_train_compare.png"
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"saved: {out}")
    else:
        print("[B8e] B8c training results not yet present; F_train_compare skipped.")

if __name__ == "__main__":
    main()
