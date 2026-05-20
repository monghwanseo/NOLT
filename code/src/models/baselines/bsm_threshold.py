import numpy as np

class BSMThresholdBaseline:
    def __init__(self, tail_window: int = 5):
        self.tail_window = tail_window
        self._train_median: float | None = None
        self._train_iqr: float | None = None

    def _score_from_pc1_window(self, pc1_window: np.ndarray) -> np.ndarray:
        diffs = np.abs(np.diff(pc1_window, axis=1))

        tw = min(self.tail_window, diffs.shape[1])
        return diffs[:, -tw:].mean(axis=1)

    def fit(self, pc1_window_train: np.ndarray, y_train: np.ndarray) -> 'BSMThresholdBaseline':
        scores = self._score_from_pc1_window(pc1_window_train)
        self._train_median = float(np.median(scores))

        q25, q75 = float(np.quantile(scores, 0.25)), float(np.quantile(scores, 0.75))
        self._train_iqr = max(q75 - q25, 1e-8)
        return self

    def predict_proba(self, pc1_window: np.ndarray) -> np.ndarray:
        if self._train_median is None:
            raise RuntimeError("call fit() before predict_proba()")
        scores = self._score_from_pc1_window(pc1_window)

        z = (scores - self._train_median) / self._train_iqr
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, pc1_window: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(pc1_window) >= threshold).astype(np.int64)
