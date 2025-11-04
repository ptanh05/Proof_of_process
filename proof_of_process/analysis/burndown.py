# proof_of_process/analysis/burndown.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _status_bin(s: str) -> str:
    s = str(s or "").lower().strip()
    if "done" in s or "completed" in s or "hoàn" in s or "xong" in s:
        return "Completed"
    if "progress" in s or "doing" in s or "đang" in s or "wip" in s:
        return "In Progress"
    return "Not Started"

def compute_burndown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["Week"] = pd.to_numeric(work["Week"], errors="coerce").fillna(1).astype(int).clip(1)
    work["StatusBin"] = work["Completion_Status"].map(_status_bin)
    agg = work.groupby("Week")["Effort_Score"].sum().reset_index(name="TotalEffort")
    done = work.loc[work["StatusBin"]=="Completed"].groupby("Week")["Effort_Score"].sum().reindex(agg["Week"]).reset_index()
    done.columns=["Week","CompletedEffort"]
    out = agg.merge(done, on="Week", how="left").fillna(0)
    out["RemainingEffort"] = (out["TotalEffort"].cumsum() - out["CompletedEffort"].cumsum()).clip(lower=0)
    return out

def compute_burnup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["Week"] = pd.to_numeric(work["Week"], errors="coerce").fillna(1).astype(int).clip(1)
    work["StatusBin"] = work["Completion_Status"].map(_status_bin)
    total_scope = work.groupby("Week")["Effort_Score"].sum().cumsum().reset_index(name="ScopeCumulative")
    completed = work.loc[work["StatusBin"]=="Completed"].groupby("Week")["Effort_Score"].sum().cumsum().reset_index(name="CompletedCumulative")
    out = total_scope.merge(completed, on="Week", how="outer").fillna(method="ffill").fillna(0)
    return out.sort_values("Week")
