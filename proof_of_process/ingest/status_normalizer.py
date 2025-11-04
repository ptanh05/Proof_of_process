import re
import pandas as pd

MAPPING = {
    "not started": ["todo","to do","backlog","open","new","planned","pending","not started","ns"],
    "in progress": ["doing","in progress","ip","working","wip","progress"],
    "completed": ["done","closed","resolved","finished","complete","completed","merged"]
}

def normalize_status(s: str) -> str:
    sl = str(s).strip().lower()
    for k, vals in MAPPING.items():
        for v in vals:
            if re.search(rf"\b{re.escape(v)}\b", sl):
                return k.title()
    return "In Progress"

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Status"] = df["Status"].map(normalize_status)
    uniq = df["Status"].dropna().unique().tolist()
    if len(uniq) < 3:  # collapse to binary if needed
        df["Status"] = df["Status"].replace({"Not Started":"In Progress"})
    return df
