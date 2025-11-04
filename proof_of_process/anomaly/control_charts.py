from __future__ import annotations
import numpy as np
import pandas as pd

def _ewma(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    mu = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
    prev = mu
    for i, v in enumerate(x):
        v = float(v) if np.isfinite(v) else prev
        prev = alpha*v + (1-alpha)*prev
        y[i] = prev
    return y

def _sigma_est(x: np.ndarray) -> float:
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0: sd = max(1e-6, np.nanmean(x)*0.25)
    return sd

def control_signals(vel: pd.Series | np.ndarray, alpha: float = 0.3, L: float = 3.0) -> dict:
    """
    Trả về: ewma, UCL/LCL, cusum_pos/neg, flags (indices)
    """
    x = np.asarray(vel, dtype=float)
    ew = _ewma(x, alpha)
    mu = np.nanmean(x)
    sd = _sigma_est(x)
    # Công thức giới hạn EWMA: sd_ew = sd * sqrt(alpha/(2-alpha))
    sd_ew = sd * np.sqrt(alpha/(2-alpha))
    UCL = ew + L*sd_ew
    LCL = np.maximum(ew - L*sd_ew, 0.0)

    # CUSUM (Page-Hinkley)
    k = 0.5*sd
    cp = np.zeros_like(x); cn = np.zeros_like(x)
    for i, v in enumerate(x):
        cp[i] = max(0.0, (cp[i-1] if i else 0.0) + (v - (mu + k)))
        cn[i] = min(0.0, (cn[i-1] if i else 0.0) + (v - (mu - k)))
    h = 5*sd
    alarm_pos = np.where(cp > h)[0].tolist()
    alarm_neg = np.where(cn < -h)[0].tolist()

    return {"ewma": ew, "mu": mu, "UCL": UCL, "LCL": LCL,
            "cusum_pos": cp, "cusum_neg": cn,
            "alarms_pos": alarm_pos, "alarms_neg": alarm_neg}
