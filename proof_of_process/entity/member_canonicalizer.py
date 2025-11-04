# proof_of_process/entity/member_canonicalizer.py
from __future__ import annotations
import re, unicodedata
from typing import Iterable, Dict
import pandas as pd

def _norm(s: str) -> str:
    s = str(s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"\s+"," ", s).lower()
    return s

def jw_similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    max_dist = max(0, max(la, lb)//2 - 1)
    match = 0
    hash_a = [False]*la
    hash_b = [False]*lb
    for i in range(la):
        start = max(0, i-max_dist)
        end = min(i+max_dist+1, lb)
        for j in range(start, end):
            if hash_b[j]: continue
            if a[i] == b[j]:
                hash_a[i] = True; hash_b[j] = True; match += 1; break
    if match == 0: return 0.0
    t = 0; point = 0
    for i in range(la):
        if not hash_a[i]: continue
        while not hash_b[point]:
            point += 1
        if a[i] != b[point]:
            t += 1
        point += 1
    t = t//2
    jaro = (match/la + match/lb + (match - t)/match)/3.0
    prefix = 0
    for i in range(min(4, la, lb)):
        if a[i] == b[i]: prefix += 1
        else: break
    return jaro + 0.1 * prefix * (1 - jaro)

def _canonical(cands):
    def quality(x: str):
        non_alpha = len(re.sub(r"[A-Za-zÀ-ỹ ]","",x))
        return (abs(len(x) - 10), non_alpha, x)
    return sorted(cands, key=quality)[0]

def dedupe_members_probabilistic(names: Iterable[str], threshold: float = 0.92) -> Dict[str,str]:
    uniq = []
    for n in names:
        n = str(n or "").strip()
        if n and n not in uniq: uniq.append(n)
    uniq_sorted = sorted(uniq, key=lambda x: (len(x), x))
    groups = []; used=set()
    for i, s in enumerate(uniq_sorted):
        if s in used: continue
        group = [s]; used.add(s)
        for t in uniq_sorted[i+1:]:
            if t in used: continue
            if jw_similarity(s, t) >= threshold:
                group.append(t); used.add(t)
        groups.append(group)
    mapping={}
    for g in groups:
        cano = _canonical(g)
        for m in g:
            mapping[m] = cano
    return mapping

def apply_member_canonicalization(df: pd.DataFrame, member_col="Assigned_Member", threshold=0.92) -> pd.DataFrame:
    if member_col not in df.columns: return df
    m = dedupe_members_probabilistic(df[member_col].astype(str).tolist(), threshold=threshold)
    out = df.copy()
    out[member_col] = out[member_col].astype(str).map(lambda x: m.get(x.strip(), x.strip()))
    if "Task_Leader" in out.columns:
        out["Task_Leader"] = out["Task_Leader"].astype(str).map(lambda x: m.get(x.strip(), x.strip()))
    return out
