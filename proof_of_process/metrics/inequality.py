from __future__ import annotations
import numpy as np
import pandas as pd

def gini(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().clip(lower=0).values
    if v.size == 0 or np.allclose(v.sum(), 0): return 0.0
    v = np.sort(v); n = v.size
    cum = np.cumsum(v)
    B = cum.sum() / (v.sum()*n)
    return float(1 + 1/n - 2*B)

def theil_T(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().clip(lower=0).values
    if v.size == 0: return 0.0
    mu = v.mean()
    if mu <= 0: return 0.0
    v = v[v>0]
    return float(np.mean((v/mu)*np.log(v/mu)))

def theil_L(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().clip(lower=0).values
    if v.size == 0: return 0.0
    mu = v.mean()
    if mu <= 0: return 0.0
    v = v[v>0]
    return float(np.mean(np.log(mu/v)))

def atkinson(x: pd.Series, eps: float = 0.5) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().clip(lower=1e-12).values
    n = v.size
    if n == 0: return 0.0
    mu = v.mean()
    if eps == 1:
        g = np.exp(np.mean(np.log(v)))
        return float(1 - g/mu)
    A = (np.mean((v**(1-eps))))**(1/(1-eps))
    return float(1 - A/mu)

def hoover(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().clip(lower=0).values
    if v.size == 0: return 0.0
    mu = v.mean()
    return float(0.5*np.mean(np.abs(v - mu))/mu) if mu>0 else 0.0

def theil_decomposition(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """
    Decomposition Theil T: Total = Between + Within (trung bình trọng số).
    """
    d = df[[group_col, value_col]].dropna().copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0).clip(lower=0)
    if d.empty or d[value_col].sum() <= 0: return {"T_total": 0.0, "T_between": 0.0, "T_within": 0.0}

    total = d[value_col].sum()
    mu = d[value_col].mean()

    T_total = theil_T(d[value_col])

    # Between
    by = d.groupby(group_col)[value_col].sum()
    mu_g = by.mean()
    T_between = theil_T(by)

    # Within (trọng số theo share nhóm)
    T_within = 0.0
    for g, sub in d.groupby(group_col):
        w = sub[value_col].sum()/total
        T_within += w * theil_T(sub[value_col])

    return {"T_total": float(T_total), "T_between": float(T_between), "T_within": float(T_within)}
