"""
Effort estimator:
- Text features (TF-IDF char + word)
- Quantile GradientBoostingRegressor for q10/q50/q90
- Conformal calibration on holdout to adjust intervals
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def _pipeline(alpha):
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)),
        ("gbm", GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42))
    ])

def fit_quantile(df: pd.DataFrame, text_col="Description", y_col="Actual_Effort"):
    data = df[[text_col, y_col]].dropna()
    if data.empty:
        return None
    X = data[text_col].astype(str).values
    y = data[y_col].astype(float).values
    Xtr, Xcal, ytr, ycal = train_test_split(X, y, test_size=0.2, random_state=42)
    q10 = _pipeline(0.10).fit(Xtr, ytr)
    q50 = _pipeline(0.50).fit(Xtr, ytr)
    q90 = _pipeline(0.90).fit(Xtr, ytr)
    # conformal: residual quantiles on calibration for q50
    y50cal = q50.predict(Xcal)
    resid = ycal - y50cal
    lo_q = np.quantile(resid, 0.10)
    hi_q = np.quantile(resid, 0.90)
    model = {"q10": q10, "q50": q50, "q90": q90, "lo": lo_q, "hi": hi_q}
    return model

def predict_quantile(model, texts):
    if model is None:
        arr = np.zeros(len(texts))
        return arr, arr, arr
    q10 = model["q10"].predict(texts)
    q50 = model["q50"].predict(texts)
    q90 = model["q90"].predict(texts)
    # conformal adjust around q50
    lo = q50 + model["lo"]
    hi = q50 + model["hi"]
    # ensure ordering
    lo = np.minimum(lo, q50)
    hi = np.maximum(hi, q50)
    return lo, q50, hi

def attach_effort(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer Actual_Effort if present; otherwise estimate
    has_actual = (df["Actual_Effort"]>0).sum() >= 20
    model = fit_quantile(df, text_col="Description", y_col="Actual_Effort") if has_actual else None
    lo, mid, hi = predict_quantile(model, df["Description"].astype(str).values)
    # Effort_Score = mid if no actual; use actual if provided
    df = df.copy()
    df["Effort_P10"] = lo
    df["Effort_P50"] = mid
    df["Effort_P90"] = hi
    df["Effort_Score"] = np.where(df["Actual_Effort"]>0, df["Actual_Effort"], df["Effort_P50"])
    return df
