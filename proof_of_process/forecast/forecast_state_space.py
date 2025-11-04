# proof_of_process/forecast/forecast_state_space.py
from __future__ import annotations
import numpy as np

def velocity_forecast_ets(vel: np.ndarray, horizon: int = 16):
    """
    Dự báo velocity bằng ETS nếu có statsmodels; trả về dict {mean, eta_p50}.
    Fallback: moving average.
    """
    arr = np.asarray(vel, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr>=0]
    if arr.size < 3:
        mu = float(np.mean(arr)) if arr.size else 0.0
        return {"mean": [mu]*horizon, "eta_p50": 0.0}

    # fallback MA
    def _ma_forecast(a, h):
        mu = float(np.mean(a[-min(5,len(a)):]))
        return [mu]*h

    try:
        import warnings
        warnings.filterwarnings("ignore")
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(arr, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        fc = fit.forecast(horizon)
        mean = [float(max(0.0, x)) for x in fc]
        # ước lượng ETA P50 cho remaining=1 đơn vị effort (chuẩn hoá)
        eta_p50 = 1.0 / (np.median(arr[-min(8,len(arr)):]) + 1e-9)
        return {"mean": mean, "eta_p50": float(max(0.0, eta_p50))}
    except Exception:
        return {"mean": _ma_forecast(arr, horizon), "eta_p50": 0.0}
