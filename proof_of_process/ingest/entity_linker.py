import pandas as pd
from .member_dedupe import split_assignees, dedupe_names

def link_entities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # explode assignees into list column
    df["Assignees_List"] = df["Assignees"].apply(split_assignees)
    # build canonical map
    all_names = [n for lst in df["Assignees_List"] for n in lst]
    canonical = dedupe_names(all_names, threshold=0.92)
    df["Assignees_List"] = df["Assignees_List"].apply(lambda lst: [canonical.get(x, x) for x in lst])
    df["Assigned_Member"] = df["Assignees_List"].apply(lambda lst: lst[0] if lst else "")
    return df
