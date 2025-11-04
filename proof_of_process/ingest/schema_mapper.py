import pandas as pd
from .schema_autodetect import guess_column

REQUIRED = ["Task","Project","Description","Assignees","Status","Week"]
OPTIONAL = ["Planned_Effort","Actual_Effort","Cost"]

def map_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    out = pd.DataFrame()
    for k in REQUIRED + OPTIONAL:
        c = guess_column(k, cols)
        out[k] = df[c] if c in df.columns else ""
    # types
    out["Planned_Effort"] = pd.to_numeric(out["Planned_Effort"], errors="coerce").fillna(0.0)
    out["Actual_Effort"]  = pd.to_numeric(out["Actual_Effort"], errors="coerce").fillna(0.0)
    out["Cost"]           = pd.to_numeric(out["Cost"], errors="coerce").fillna(0.0)
    out["Task"] = out["Task"].astype(str)
    out["Project"] = out["Project"].astype(str)
    out["Description"] = out["Description"].astype(str)
    out["Assignees"] = out["Assignees"].astype(str)
    out["Status"] = out["Status"].astype(str)
    # Chuẩn hoá cột Week thành tuần (string) an toàn
    w = pd.to_datetime(out["Week"], errors="coerce")
    # Nếu toàn NaT, fallback về tuần hiện tại để pipeline không vỡ
    if w.isna().all():
        w = pd.Series(pd.Timestamp.today(), index=out.index)
    out["Week"] = pd.PeriodIndex(w, freq="W").astype(str)
    out["Week"] = out["Week"].ffill()

    return out
