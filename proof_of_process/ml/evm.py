import pandas as pd
import numpy as np

def compute_evm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires columns: Week (YYYY-WW), Status, Planned_Effort, Effort_Score, Cost
    EV: cumulative effort of completed tasks
    PV: cumulative planned effort by week (fallback: equal spread)
    AC: cumulative Cost or Actual_Effort (fallback: Effort_Score)
    """
    d = df.copy()
    d["Week"] = pd.PeriodIndex(d["Week"], freq="W").to_timestamp()
    d["is_completed"] = (d["Status"]=="Completed").astype(int)
    weekly = d.groupby("Week").agg(
        ev_done=("Effort_Score", lambda s: s[d.loc[s.index,"is_completed"]==1].sum()),
        pv=("Planned_Effort", "sum"),
        ac=("Cost","sum"),
        n_done=("is_completed","sum")
    ).sort_index()
    # PV fallback: if zero, approximate = total planned evenly over horizon
    if (weekly["pv"]==0).all():
        total = d["Planned_Effort"].sum()
        if total==0:
            total = d["Effort_Score"].sum()
        if len(weekly)>0:
            weekly["pv"] = total/len(weekly)
    weekly["EV"] = weekly["ev_done"].cumsum()
    weekly["PV"] = weekly["pv"].cumsum()
    if weekly["ac"].sum()==0:
        # fallback AC from effort of done
        weekly["ac"] = weekly["ev_done"]
    weekly["AC"] = weekly["ac"].cumsum()
    weekly["SPI"] = np.where(weekly["PV"]>0, weekly["EV"]/weekly["PV"], 1.0)
    weekly["CPI"] = np.where(weekly["AC"]>0, weekly["EV"]/weekly["AC"], 1.0)
    weekly["Velocity"] = weekly["ev_done"]
    return weekly.reset_index()
