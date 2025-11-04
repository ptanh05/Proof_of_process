"""
Trello connector (minimal). Two modes:
1) Live API (requires env TRELLO_KEY, TRELLO_TOKEN) – TODO: implement fetch if needed.
2) Offline: read a CSV export you already have.
Returns normalized DataFrame with columns:
[Task, Project, Description, Assignees, Status, Week, Planned_Effort, Actual_Effort, Cost]
"""
import os
import pandas as pd
from dateutil.parser import parse as dtparse

def from_csv(path: str, project_name: str = "TrelloProject") -> pd.DataFrame:
    df = pd.read_csv(path)
    # heuristic mapping
    cols = {c.lower(): c for c in df.columns}
    task = cols.get("name") or cols.get("task") or list(df.columns)[0]
    desc = cols.get("desc") or cols.get("description")
    listcol = cols.get("list") or cols.get("status") or cols.get("column")
    assg = cols.get("members") or cols.get("assignees") or cols.get("assigned_to")
    due = cols.get("due") or cols.get("deadline") or cols.get("date")
    out = pd.DataFrame()
    out["Task"] = df[task].astype(str)
    out["Project"] = project_name
    out["Description"] = df[desc] if desc else ""
    out["Assignees"] = df[assg] if assg else ""
    out["Status"] = df[listcol] if listcol else "In Progress"
    if due and df[due].notna().any():
        out["Week"] = pd.to_datetime(df[due].apply(lambda x: dtparse(str(x)) if pd.notna(x) else pd.NaT)).dt.to_period("W").astype(str)
    else:
        out["Week"] = pd.Timestamp.today().to_period("W").astype(str)
    for c in ["Planned_Effort","Actual_Effort","Cost"]:
        out[c] = 0.0
    return out
