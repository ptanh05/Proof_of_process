import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

def anomaly_velocity(weekly_df: pd.DataFrame) -> pd.DataFrame:
    if weekly_df.empty:
        return weekly_df.assign(Anomaly=0)
    X = weekly_df[["Velocity"]].fillna(0.0).values
    clf = IsolationForest(random_state=42, contamination=0.1)
    y = clf.fit_predict(X)
    return weekly_df.assign(Anomaly=(y==-1).astype(int))
