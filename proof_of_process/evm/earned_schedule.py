from __future__ import annotations
import numpy as np
import pandas as pd

def earned_schedule(burnup: pd.DataFrame, time_col: str = "Week") -> dict:
    """
    Tính Earned Schedule (ES), Schedule Variance in time SV(t) và SPI(t).
    burnup: DataFrame có cột time_col, ScopeCumulative, CompletedCumulative.
    """
    df = burnup.copy()
    df = df.sort_values(time_col)
    t = df[time_col].astype(float).values
    pv = df["ScopeCumulative"].astype(float).values
    ev = df["CompletedCumulative"].astype(float).values
    if len(t) < 2 or np.nanmax(ev) <= 0:
        return {}

    # ES = thời điểm PV == EV hiện tại (nội suy tuyến tính trên PV theo t)
    EV_now = ev[-1]
    # tìm đoạn [i,i+1] sao cho PV_i <= EV_now <= PV_{i+1}
    idx = np.searchsorted(pv, EV_now, side="right") - 1
    idx = np.clip(idx, 0, len(pv)-2)
    pv0, pv1 = pv[idx], pv[idx+1]
    t0,  t1  = t[idx],  t[idx+1]
    if pv1 - pv0 <= 1e-9:
        ES = float(t0)
    else:
        w = (EV_now - pv0) / (pv1 - pv0)
        ES = float(t0 + w*(t1 - t0))

    AT = float(t[-1])  # Actual Time tới hiện tại
    SV_t = ES - AT
    SPI_t = ES / AT if AT > 0 else np.nan
    return {"ES": ES, "AT": AT, "SV_t": SV_t, "SPI_t": SPI_t}
