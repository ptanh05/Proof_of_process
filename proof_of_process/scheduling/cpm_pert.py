from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set

# ----------------- Helpers -----------------
def _parse_deps(s):
    if isinstance(s, list): return [str(x).strip() for x in s if str(x).strip()]
    s = str(s or "").strip()
    if not s: return []
    for sep in [";", ",", "|", "/", "+"]:
        s = s.replace(sep, " ")
    return [t.strip() for t in s.split() if t.strip()]

def _duration_row(r: pd.Series) -> tuple[float,float,float]:
    def _getnum(x):
        try: return float(x)
        except: return np.nan
    a = _getnum(r.get("Opt_Duration", np.nan))
    m = _getnum(r.get("Most_Duration", np.nan))
    b = _getnum(r.get("Pess_Duration", np.nan))
    if not np.isfinite(a) or not np.isfinite(m) or not np.isfinite(b):
        e = _getnum(r.get("Effort_Score", np.nan))
        if not np.isfinite(e): e = 1.0
        a = 0.8*e; m = e; b = 1.25*e
    return float(a), float(m), float(b)

# ----------------- Tarjan SCC -----------------
def _scc_tarjan(nodes: List[str], edges: Dict[str, List[str]]) -> List[List[str]]:
    idx = 0
    index, lowlink, onstack = {}, {}, {}
    S = []
    comps = []

    def strongconnect(v):
        nonlocal idx
        index[v] = idx; lowlink[v] = idx; idx += 1
        S.append(v); onstack[v] = True
        for w in edges.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif onstack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = S.pop()
                onstack[w] = False
                comp.append(w)
                if w == v: break
            comps.append(comp)

    for v in nodes:
        if v not in index:
            strongconnect(v)
    return comps  # list of components (each is list of nodes)

# ----------------- CPM (strict) -----------------
def compute_cpm(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    CPM nghiêm ngặt: yêu cầu DAG. Nếu có chu trình sẽ raise ValueError.
    """
    gpred = defaultdict(list)
    gsucc = defaultdict(list)
    tasks = []
    dur = {}
    order = {}  # giữ thứ tự input
    for i, r in df.iterrows():
        tid = str(r.get("Task","")).strip() or f"T{i}"
        if tid not in order: order[tid] = i
        preds = [p for p in _parse_deps(r.get("Depends_On","")) if p != tid]
        a, m, b = _duration_row(r)
        d = m
        tasks.append(tid); dur[tid] = max(0.0, float(d))
        for p in preds:
            gsucc[p].append(tid); gpred[tid].append(p)

    # topo sort (Kahn)
    indeg = {t: len(gpred[t]) for t in tasks}
    q = deque([t for t in tasks if indeg[t]==0])
    topo = []
    while q:
        u = q.popleft(); topo.append(u)
        for v in gsucc[u]:
            indeg[v] -= 1
            if indeg[v]==0: q.append(v)
    if len(topo)!=len(tasks):
        # kèm thêm danh sách chu trình cho dễ sửa dữ liệu
        comps = _scc_tarjan(tasks, gsucc)
        cycles = [c for c in comps if len(c)>1]
        msg = "Cycle detected in Depends_On"
        if cycles:
            msg += f" — cycles: {[' -> '.join(c) for c in cycles[:3]]}"
        raise ValueError(msg)

    ES, EF = {}, {}
    for u in topo:
        es = 0.0 if not gpred[u] else max(EF[p] for p in gpred[u])
        ef = es + dur[u]
        ES[u], EF[u] = es, ef
    proj_duration = max(EF.values()) if EF else 0.0

    LS, LF = {}, {}
    for u in reversed(topo):
        lf = proj_duration if not gsucc[u] else min(LS[v] for v in gsucc[u])
        ls = lf - dur[u]
        LS[u], LF[u] = ls, lf

    rows=[]
    for t in topo:
        slack = LS[t] - ES[t]
        rows.append({"Task": t, "ES": ES[t], "EF": EF[t], "LS": LS[t], "LF": LF[t],
                     "Duration": dur[t], "Slack": slack, "Critical": bool(abs(slack) < 1e-9)})
    out = pd.DataFrame(rows).sort_values("ES", kind="stable")
    return out, {"duration": proj_duration, "critical_count": int(out["Critical"].sum())}

# ----------------- CPM (relaxed, cycle-tolerant) -----------------
def compute_cpm_relaxed(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    CPM 'relaxed': gom mỗi SCC (chu trình) thành 1 siêu-nút có Duration = SUM durations,
    tính CPM trên đồ thị đã gom (luôn là DAG), rồi TRẢI lại các task nội bộ theo thứ tự input.
    """
    # build graph
    gpred = defaultdict(list)
    gsucc = defaultdict(list)
    tasks = []
    dur = {}
    order = {}
    for i, r in df.iterrows():
        tid = str(r.get("Task","")).strip() or f"T{i}"
        if tid not in order: order[tid] = i
        preds = [p for p in _parse_deps(r.get("Depends_On","")) if p != tid]
        a, m, b = _duration_row(r)
        d = m
        tasks.append(tid); dur[tid] = max(0.0, float(d))
        for p in preds:
            gsucc[p].append(tid); gpred[tid].append(p)

    # SCC
    comps = _scc_tarjan(tasks, gsucc)  # list[list[task]]
    comp_id = {}
    for cid, comp in enumerate(comps):
        for t in comp: comp_id[t] = cid

    # condense graph
    comp_nodes = list(range(len(comps)))
    comp_dur = {cid: float(sum(dur[t] for t in comps[cid])) for cid in comp_nodes}
    comp_preds = defaultdict(set)
    comp_succs = defaultdict(set)
    for u in tasks:
        cu = comp_id[u]
        for v in gsucc[u]:
            cv = comp_id[v]
            if cu != cv:
                comp_succs[cu].add(cv)
                comp_preds[cv].add(cu)

    # topo on condensed DAG
    indeg = {c: len(comp_preds[c]) for c in comp_nodes}
    q = deque([c for c in comp_nodes if indeg[c]==0])
    topo = []
    while q:
        u = q.popleft(); topo.append(u)
        for v in comp_succs[u]:
            indeg[v] -= 1
            if indeg[v]==0: q.append(v)

    # forward/backward on components
    ES_c, EF_c = {}, {}
    for u in topo:
        es = 0.0 if not comp_preds[u] else max(EF_c[p] for p in comp_preds[u])
        ef = es + comp_dur[u]
        ES_c[u], EF_c[u] = es, ef
    proj_duration = max(EF_c.values()) if EF_c else 0.0

    LS_c, LF_c = {}, {}
    for u in reversed(topo):
        lf = proj_duration if not comp_succs[u] else min(LS_c[v] for v in comp_succs[u])
        ls = lf - comp_dur[u]
        LS_c[u], LF_c[u] = ls, lf

    # expand back to tasks: tuần tự theo thứ tự input trong cùng SCC
    rows=[]
    for cid, comp in enumerate(comps):
        comp_sorted = sorted(comp, key=lambda t: order.get(t, 0))
        offset = 0.0
        for t in comp_sorted:
            es = ES_c[cid] + offset
            ef = es + dur[t]
            # slack của task = slack của component (xấp xỉ hợp lý)
            slack = LS_c[cid] - ES_c[cid]
            rows.append({
                "Task": t, "ES": es, "EF": ef, "LS": es + slack, "LF": ef + slack,
                "Duration": dur[t], "Slack": slack, "Critical": bool(abs(slack) < 1e-9),
                "Cycle_Group": f"SCC_{cid}" if len(comp)>1 else ""
            })
            offset += dur[t]

    out = pd.DataFrame(rows).sort_values(["ES","Task"], kind="stable")
    crit_cnt = int(out["Critical"].sum())
    # danh sách chu trình để export/log
    cycles = [c for c in comps if len(c)>1]
    return out, {
        "duration": proj_duration,
        "critical_count": crit_cnt,
        "cycles": cycles
    }

# ----------------- PERT Monte Carlo (giữ nguyên) -----------------
def _beta_pert(a,m,b, size=1):
    a, m, b = float(a), float(m), float(b)
    if b <= a: return np.full(size, m)
    alpha = 1 + 4*(m-a)/(b-a)
    beta  = 1 + 4*(b-m)/(b-a)
    x = np.random.beta(alpha, beta, size=size)
    return a + x*(b-a)

def monte_carlo_pert(df: pd.DataFrame, samples: int = 2000) -> dict:
    durations = []
    for _, r in df.iterrows():
        a,m,b = _duration_row(r)
        durations.append(_beta_pert(a,m,b, size=samples))
    if not durations: return {}
    # baseline CPM strict; nếu fail thì dùng relaxed để lấy critical set
    try:
        cpm_df, _ = compute_cpm(df)
        crit_tasks = set(cpm_df.loc[cpm_df["Critical"], "Task"].tolist())
    except Exception:
        cpm_df, _ = compute_cpm_relaxed(df)
        crit_tasks = set(cpm_df.loc[cpm_df["Critical"], "Task"].tolist())
    dur_arr = np.vstack(durations)  # [n_task, samples]
    # trọng số: 1 với task trên đường găng (xấp xỉ)
    # nếu không rõ thì dùng tổng
    if crit_tasks:
        weights = np.array([1.0 if str(df.iloc[i].get("Task","")).strip() in crit_tasks else 0.0
                            for i in range(len(df))], dtype=float).reshape(-1,1)
        tot = (dur_arr * weights).sum(axis=0)
        tot = np.maximum(tot, dur_arr.max(axis=0))
    else:
        tot = dur_arr.sum(axis=0)
    return {"duration_samples": tot, "p50": float(np.quantile(tot,0.5))}
