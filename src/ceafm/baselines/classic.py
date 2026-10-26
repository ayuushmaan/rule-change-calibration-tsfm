"""Simple baselines."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def naive_last(y_hist):
    return y_hist[-1] if len(y_hist) else 0.0


def moving_avg(y_hist, w=5):
    if len(y_hist) == 0:
        return 0.0
    return np.mean(y_hist[-w:])


def ridge_baseline(X_tr, y_tr, X_te):
    m = Ridge(alpha=1.0)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)
