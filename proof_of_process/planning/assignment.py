from __future__ import annotations
import numpy as np
import pandas as pd

def _cosine(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

def _vec_from_labels(labels: list[str], buckets: list[str]) -> np.ndarray:
    vec = np.zeros(len(buckets), dtype=float)
    for lab in labels:
        try:
            k = buckets.index(lab)
            vec[k] += 1.0
        except ValueError:
            continue
    return vec

def suggest_assignments(df: pd.DataFrame,
                        member_profiles: dict[str, pd.Series],
                        topn: int = 50) -> pd.DataFrame:
    """
    Scoring đơn giản: cosine(skill_vector(member), yêu cầu task).
    Yêu cầu task ~ count các bucket xuất hiện trong Skills (đã infer).
    Nếu không có Skills, fallback Effort_Score -> General.
    """
    # bucket order từ profiles
    buckets = list(next(iter(member_profiles.values())).index) if member_profiles else []
    if not buckets: return pd.DataFrame()

    # vec người
    mem_vec = {m: np.array(member_profiles[m].values, dtype=float) for m in member_profiles}

    rows=[]
    for _, r in df.iterrows():
        task = str(r.get("Task","")).strip()
        eff  = float(pd.to_numeric(r.get("Effort_Score", 1.0), errors="coerce") or 1.0)
        # lấy nhãn bucket từ Skills (là label sau mapping)
        labels = []
        s = str(r.get("Skills","")).strip()
        if s.startswith("["):
            try:
                import json
                for x in json.loads(s):
                    labels.append(str(x))
            except: pass
        if not labels:
            labels = ["General"]
        v_task = _vec_from_labels(labels, buckets)
        v_task = (v_task / (np.linalg.norm(v_task) or 1.0)) * eff

        for m, v_mem in mem_vec.items():
            score = _cosine(v_mem, v_task)
            rows.append({"Task": task, "Member": m, "Score": score})

    out = pd.DataFrame(rows)
    out = out.sort_values(["Task","Score"], ascending=[True, False])
    return out.groupby("Task").head(3).sort_values("Score", ascending=False).head(topn).reset_index(drop=True)
