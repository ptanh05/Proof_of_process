from __future__ import annotations
import numpy as np
import pandas as pd

def _nz(x, fallback=1.0):
    try:
        v = float(x)
        return v if np.isfinite(v) and v>0 else fallback
    except:
        return fallback

def compute_wsjf(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Heuristic fallback nếu thiếu cột
    if "Job_Size" not in d.columns or d["Job_Size"].isna().all():
        d["Job_Size"] = d.get("Effort_Score", 1.0)
    for c,fb in [("Business_Value",1.0), ("Time_Criticality",1.0), ("Risk_Reduction",1.0), ("Job_Size",1.0)]:
        d[c] = pd.to_numeric(d.get(c, fb), errors="coerce").fillna(fb)
    d["WSJF"] = (d["Business_Value"] + d["Time_Criticality"] + d["Risk_Reduction"]) / d["Job_Size"].clip(lower=0.1)
    cols = ["Project","Task","WSJF","Business_Value","Time_Criticality","Risk_Reduction","Job_Size"]
    cols = [c for c in cols if c in d.columns]
    return d.sort_values("WSJF", ascending=False)[cols]

def compute_rice(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Effort" not in d.columns or d["Effort"].isna().all():
        d["Effort"] = d.get("Job_Size", d.get("Effort_Score", 1.0))
    for c,fb in [("Reach",1.0),("Impact",1.0),("Confidence",0.7),("Effort",1.0)]:
        d[c] = pd.to_numeric(d.get(c, fb), errors="coerce").fillna(fb)
    d["RICE"] = (d["Reach"] * d["Impact"] * d["Confidence"]) / d["Effort"].clip(lower=0.1)
    cols = ["Project","Task","RICE","Reach","Impact","Confidence","Effort"]
    cols = [c for c in cols if c in d.columns]
    return d.sort_values("RICE", ascending=False)[cols]
