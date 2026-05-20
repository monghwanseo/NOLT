REAL_BEST = {
    "bsm_threshold":   {"tail_window": 3},
    "garch":           {"p": 2, "q": 1},
    "xgboost":         {"n_estimators": 30, "max_depth": 7, "learning_rate": 0.1, "reg_lambda": 1.0},
    "lstm_single":     {"hidden_dim": 64, "n_layers": 2, "dropout": 0.1, "lr": 1e-4,
                        "batch": 32, "weight_decay": 1e-3, "patience": 20, "max_epochs": 120},
    "nolt_full":       {"d_model": 128, "n_layers": 2, "dropout": 0.2, "n_heads": 4,
                        "lr": 3e-4, "batch": 32, "weight_decay": 1e-3,
                        "patience": 20, "max_epochs": 120},
    "nolt_snap":       {"d_model": 32, "n_layers": 3, "dropout": 0.3, "n_heads": 4,
                        "lr": 3e-4, "batch": 32, "weight_decay": 1e-3,
                        "patience": 20, "max_epochs": 120},
}

SYNTH_BEST_HESTON = {
    "bsm_threshold":   {"tail_window": 15},
    "xgboost":         {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "reg_lambda": 1.0},
    "lstm_single":     {"hidden_dim": 64, "n_layers": 1, "dropout": 0.2, "lr": 5e-4,
                        "batch": 64, "weight_decay": 1e-3, "patience": 10, "max_epochs": 40},
    "nolt_full":       {"d_model": 64, "n_layers": 2, "dropout": 0.2, "n_heads": 4,
                        "lr": 3e-4, "batch": 64, "weight_decay": 1e-3,
                        "patience": 10, "max_epochs": 40},
    "nolt_snap":       {"d_model": 32, "n_layers": 3, "dropout": 0.3, "n_heads": 4,
                        "lr": 3e-4, "batch": 64, "weight_decay": 1e-3,
                        "patience": 10, "max_epochs": 40},
}

SYNTH_BEST_BATES = dict(SYNTH_BEST_HESTON)

SEED = 2026
THRESHOLD_QUANTILE_REAL = 0.85
THRESHOLD_QUANTILE_SYNTH = 0.90
LOOKBACK = 60
TRAIN_VAL_TEST = (0.70, 0.15, 0.15)
