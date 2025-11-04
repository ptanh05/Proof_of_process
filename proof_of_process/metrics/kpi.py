# proof_of_process/metrics/kpi.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_mean(x):
    x = pd.to_numeric(pd.Series(list(x)), errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else 0.0

def build_kpis(data: pd.DataFrame,
               weekly: pd.DataFrame | None,
               contrib: pd.DataFrame | None,
               remaining_effort: float,
               eta: dict) -> dict:
    k = {}
    k["Projects"] = int(data["Project"].nunique())
    k["Members"]  = int(data["Assigned_Member"].nunique())
    k["Effort_Total"] = float(data["Effort_Score"].sum())
    k["Effort_Remaining"] = float(remaining_effort)

    if weekly is not None and not weekly.empty:
        for c in ["SPI","CPI","Velocity"]:
            if c in weekly.columns:
                k[c+"_mean"] = _safe_mean(weekly[c])
        last = weekly.sort_values("Week").tail(1)
        if not last.empty:
            k["SPI_last"] = float(last["SPI"].values[0]) if "SPI" in last.columns else 1.0
            k["CPI_last"] = float(last["CPI"].values[0]) if "CPI" in last.columns else 1.0
            k["Velocity_last"] = float(last["Velocity"].values[0]) if "Velocity" in last.columns else 0.0
    else:
        k.update({"SPI_mean":1.0,"CPI_mean":1.0,"Velocity_mean":0.0,"SPI_last":1.0,"CPI_last":1.0,"Velocity_last":0.0})

    k["ETA_P10"] = float(eta.get("p10",0.0))
    k["ETA_P50"] = float(eta.get("p50",0.0))
    k["ETA_P90"] = float(eta.get("p90",0.0))

    # Risk mặc định (nếu không có module cũ)
    spi = k.get("SPI_last",1.0)
    vel = k.get("Velocity_mean",0.0)
    rem = k["Effort_Remaining"]
    # heuristic: SPI<0.9, velocity thấp & remaining lớn -> rủi ro cao hơn
    risk_score = 50
    if spi < 0.9: risk_score -= 15
    if vel < max(1.0, 0.1* k["Effort_Total"] ): risk_score -= 10
    if rem > 0.5 * k["Effort_Total"]: risk_score -= 10
    risk_score = int(np.clip(risk_score, 0, 100))
    level = "Low" if risk_score >= 70 else "Medium" if risk_score >= 40 else "High"
    k["Risk_Score"] = risk_score
    k["Risk_Level"] = level

    return k
