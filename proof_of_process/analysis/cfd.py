# proof_of_process/analysis/cfd.py
from __future__ import annotations
import pandas as pd

def _bin(s: str) -> str:
    s = str(s or "").lower().strip()
    if "done" in s or "completed" in s or "hoàn" in s or "xong" in s:
        return "Completed"
    if "progress" in s or "doing" in s or "đang" in s or "wip" in s:
        return "In Progress"
    return "Not Started"

def compute_cfd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    w = df.copy()
    w["Week"] = pd.to_numeric(w["Week"], errors="coerce").fillna(1).astype(int).clip(1)
    w["StatusBin"] = w["Completion_Status"].map(_bin)
    pivot = w.pivot_table(index="Week", columns="StatusBin", values="Effort_Score", aggfunc="sum", fill_value=0)
    # cumulative theo thời gian
    out = pivot.cumsum().reset_index()
    # bảo đảm có đủ cột
    for col in ["Not Started","In Progress","Completed"]:
        if col not in out.columns:
            out[col] = 0.0
    return out[["Week","Not Started","In Progress","Completed"]].sort_values("Week")
