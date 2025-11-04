from __future__ import annotations
import numpy as np
import pandas as pd

def _skills_to_rows(skills_cell):
    import json
    if isinstance(skills_cell, list): return skills_cell
    s = str(skills_cell or "").strip()
    if s.startswith("["):
        try: return json.loads(s)
        except: return []
    return []

def build_bipartite(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trả DataFrame cạnh (Member, Task, Effort) từ bảng đã expand theo member.
    """
    d = expanded_df.copy()
    d["Effort_Score"] = pd.to_numeric(d.get("Effort_Score", 1.0), errors="coerce").fillna(1.0)
    d["Assigned_Member"] = d["Assigned_Member"].astype(str)
    d["Task"] = d["Task"].astype(str)
    return d[["Assigned_Member","Task","Effort_Score"]].rename(columns={
        "Assigned_Member":"Member", "Effort_Score":"Effort"
    })

def metrics_bus_factor(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Chỉ số rủi ro người-khóa (bus-factor):
    - degree_tasks: số task tham gia
    - solo_tasks: số task chỉ có 1 người được assign
    - solo_share: solo_tasks / degree_tasks
    - key_person_index: 0.7*solo_share + 0.3*(degree_tasks / max_degree)
    """
    d = expanded_df.copy()
    d["Member"] = d["Assigned_Member"].astype(str)
    # số người/ task
    members_per_task = d.groupby("Task")["Member"].nunique()
    d = d.merge(members_per_task.rename("mpt"), left_on="Task", right_index=True, how="left")
    deg = d.groupby("Member")["Task"].nunique().rename("degree_tasks")
    solo = d[d["mpt"]==1].groupby("Member")["Task"].nunique().rename("solo_tasks")
    out = pd.concat([deg, solo], axis=1).fillna(0.0)
    out["solo_share"] = np.where(out["degree_tasks"]>0, out["solo_tasks"]/out["degree_tasks"], 0.0)
    maxdeg = max(float(out["degree_tasks"].max()), 1.0)
    out["key_person_index"] = 0.7*out["solo_share"] + 0.3*(out["degree_tasks"]/maxdeg)
    return out.reset_index().rename(columns={"index":"Member"}).sort_values("key_person_index", ascending=False)
