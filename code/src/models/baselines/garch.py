import warnings

import numpy as np
from arch import arch_model
from scipy.stats import norm

class GARCHBaseline:
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self._res = None
        self._dpc1_train: np.ndarray | None = None
        self._train_dpc1_std: float | None = None

    def fit(self, dpc1_train: np.ndarray) -> 'GARCHBaseline':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = arch_model(dpc1_train * 100.0,
                                vol='GARCH', p=self.p, q=self.q,
                                rescale=False)
            self._res = model.fit(disp='off', show_warning=False)
        self._dpc1_train = np.asarray(dpc1_train, dtype=float)
        self._train_dpc1_std = float(np.std(dpc1_train) + 1e-12)
        return self

    def _cond_sigma_from_window(self, pc1_window: np.ndarray) -> np.ndarray:
        if self._res is None or self._dpc1_train is None:
            raise RuntimeError("call fit() before predicting")
        params = self._res.params
        omega = float(params.get('omega', 0.0))
        alpha = float(params.get('alpha[1]', 0.0))
        beta = float(params.get('beta[1]', 0.0))

        scaled_train = self._dpc1_train * 100.0
        long_var = float(np.var(scaled_train))

        sig2_init = long_var

        diffs = np.diff(pc1_window, axis=1) * 100.0
        N_s, L = diffs.shape

        sig2_pred = np.zeros(N_s)
        for i in range(N_s):

            sig2 = sig2_init
            for k in range(L):
                sig2 = omega + alpha * (diffs[i, k] ** 2) + beta * sig2

            sig2_pred[i] = sig2

        return np.sqrt(np.maximum(sig2_pred, 1e-12)) / 100.0

    def predict_proba(self, pc1_window: np.ndarray, threshold: float) -> np.ndarray:
        sigma = self._cond_sigma_from_window(pc1_window)

        z = threshold / np.maximum(sigma, 1e-12)
        return 2.0 * (1.0 - norm.cdf(z))

    def predict(self, pc1_window: np.ndarray, threshold: float, decision_p: float = 0.5) -> np.ndarray:
        return (self.predict_proba(pc1_window, threshold) >= decision_p).astype(np.int64)
