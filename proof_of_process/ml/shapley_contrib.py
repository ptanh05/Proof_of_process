"""
Shapley contribution % (approx):
- Value of a task = Effort_Score (or weighted)
- Players = assignees on that task
- Sample permutations to estimate marginal contributions
"""
import numpy as np, pandas as pd, random

def shapley_for_task(members, task_value, samples=200, seed=42):
    if not members: return {}
    if len(members)==1: return {members[0]: task_value}
    random.seed(seed)
    contrib = {m:0.0 for m in members}
    for _ in range(samples):
        perm = members[:]
        random.shuffle(perm)
        acc = 0.0
        seen = set()
        for m in perm:
            # marginal: full value equally split among all participants in this simple cooperative game
            # More sophisticated: could weight by role/time, but we keep unbiased split per coalition
            prev = acc
            # coalition value when m joins: we assume value grows to full only when all present;
            # to avoid undercredit, use proportional by number joined
            k = len(seen)+1
            v = task_value * (k/len(members))
            contrib[m] += (v - prev)
            acc = v
            seen.add(m)
    return contrib

def contribution_percent(df: pd.DataFrame, value_col="Effort_Score", samples=200) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        members = r.get("Assignees_List", [])
        val = float(r.get(value_col, 0.0))
        if not members or val<=0:
            continue
        for m, v in shapley_for_task(members, val, samples=samples).items():
            rows.append((r["Project"], m, v))
    if not rows:
        return pd.DataFrame(columns=["Project","Member","Contribution_%"])
    agg = pd.DataFrame(rows, columns=["Project","Member","Value"]).groupby(["Project","Member"])["Value"].sum().reset_index()
    agg["Contribution_%"] = agg.groupby("Project")["Value"].transform(lambda x: 100*x/x.sum() if x.sum()>0 else 0.0)
    return agg.sort_values(["Project","Contribution_%"], ascending=[True,False])
