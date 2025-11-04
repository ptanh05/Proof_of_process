"""
Heuristic + logistic-like risk score:
Features: last SPI, SPI trend, overdue rate, scope creep, Gini of contributions
Output: Risk_Level (Low/Medium/High) + Score_0_100
"""
import numpy as np, pandas as pd

def gini(array):
    x = np.array(array, dtype=float)
    x = x[x>=0]
    if len(x)==0: return 0.0
    if x.sum()==0: return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    g = (n+1 - 2*(cum.sum()/cum[-1]))/n
    return float(g)

def risk_from_kpis(weekly_evm: pd.DataFrame, tasks_df: pd.DataFrame, contrib_df: pd.DataFrame):
    if weekly_evm.empty:
        return {"Risk_Level":"Medium","Score_0_100":50.0, "Notes":"No EVM history"}
    spi_last = weekly_evm["SPI"].iloc[-1]
    spi_trend = weekly_evm["SPI"].tail(4).diff().mean() if len(weekly_evm)>=4 else 0.0
    # overdue: tasks not completed with Week < latest
    latest_w = weekly_evm["Week"].max()
    overdue = tasks_df[(tasks_df["Status"]!="Completed") & (pd.PeriodIndex(tasks_df["Week"], freq="W").to_timestamp()<latest_w)]
    overdue_rate = len(overdue)/max(1,len(tasks_df))
    # scope creep: new tasks added last 3 weeks vs earlier average
    week_counts = tasks_df.groupby(tasks_df["Week"]).size().sort_index()
    recent = week_counts.tail(3).mean() if len(week_counts)>=3 else week_counts.mean()
    prev = week_counts.iloc[:-3].mean() if len(week_counts)>3 else week_counts.mean()
    creep = (recent - prev)/max(prev,1e-6)
    # gini
    g = 0.0
    if not contrib_df.empty:
        g = gini(contrib_df["Contribution_%"].values)
    # logistic-ish score
    z = -1.5*(1-spi_last) - 1.0*overdue_rate - 0.5*max(creep,0) - 0.3*g + 0.8*spi_trend
    prob_ok = 1/(1+np.exp(-z))
    score = float(100*prob_ok)
    if score>=70: lvl="Low"
    elif score>=40: lvl="Medium"
    else: lvl="High"
    return {"Risk_Level": lvl, "Score_0_100": score,
            "Notes": f"SPI={spi_last:.2f}, overdue={overdue_rate:.2f}, creep={creep:.2f}, Gini={g:.2f}, trend={spi_trend:.3f}"}
