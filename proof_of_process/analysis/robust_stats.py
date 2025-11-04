# proof_of_process/analysis/robust_stats.py
from __future__ import annotations
import numpy as np
import pandas as pd

def winsorize_series(s: pd.Series, lower=0.05, upper=0.95) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    ql, qu = x.quantile(lower), x.quantile(upper)
    return x.clip(lower=ql, upper=qu)

def huber_mean(x, c=1.345, eps=1e-6, max_iter=50):
    """Ước lượng Huber location (robust)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    mu = np.median(x)
    for _ in range(max_iter):
        r = x - mu
        s = np.median(np.abs(r)) + eps
        w = np.minimum(1.0, c * s / (np.abs(r) + eps))
        mu_new = np.sum(w * x) / np.sum(w)
        if abs(mu_new - mu) < eps:
            break
        mu = mu_new
    return float(mu)
