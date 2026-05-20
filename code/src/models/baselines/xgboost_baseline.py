import numpy as np
import xgboost as xgb

from src.data import config as cfg

def build_xgboost_features(R_window: np.ndarray, pc1_window: np.ndarray) -> np.ndarray:
    N_s, T, N_opt = R_window.shape

    feat_pc1 = pc1_window

    feat_dpc1 = np.abs(np.diff(pc1_window, axis=1))

    rolls = []
    for w in (5, 10, 20):
        w = min(w, T)
        sl = pc1_window[:, -w:]
        rolls.append(sl.mean(axis=1, keepdims=True))
        rolls.append(sl.std(axis=1, keepdims=True))
        rolls.append(np.abs(sl).max(axis=1, keepdims=True))
    feat_rolls = np.concatenate(rolls, axis=1)

    feat_last_R = R_window[:, -1, :]

    return np.concatenate([feat_pc1, feat_dpc1, feat_rolls, feat_last_R], axis=1)

class XGBoostBaseline:
    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: int = 5,
                 learning_rate: float = 0.05,
                 subsample: float = 0.9,
                 colsample_bytree: float = 0.8,
                 min_child_weight: float = 1.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 seed: int = cfg.SEED,
                 ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=-1,
        )
        self.model: xgb.XGBClassifier | None = None

    def fit(self, X_features: np.ndarray, y: np.ndarray,
            X_val_features: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            early_stopping_rounds: int | None = 30) -> 'XGBoostBaseline':

        pos_rate = float(y.mean())
        scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)
        params = dict(self.params)
        params['scale_pos_weight'] = scale_pos_weight
        if early_stopping_rounds and X_val_features is not None:
            params['early_stopping_rounds'] = early_stopping_rounds

        self.model = xgb.XGBClassifier(**params)
        if X_val_features is not None and y_val is not None:
            self.model.fit(X_features, y, eval_set=[(X_val_features, y_val)], verbose=False)
        else:
            self.model.fit(X_features, y, verbose=False)
        return self

    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X_features)[:, 1]

    def predict(self, X_features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_features) >= threshold).astype(np.int64)
