# proof_of_process/pipeline/orchestrator.py
from __future__ import annotations
import os, json, math, warnings, importlib, types
from typing import Dict, Any, Tuple, List, Iterable
import numpy as np
import pandas as pd

# ============ Logger ============
try:
    from ..utils.logging_utils import get_logger
except Exception:
    def get_logger():
        import logging
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)s - %(message)s",
                            force=True)
        return logging.getLogger("pipeline")
log = get_logger()

# ============ Robust imports ============
def _try_import(module_name: str):
    try:
        mod = importlib.import_module(module_name)
        log.info(f"[import] loaded: {module_name}")
        return mod
    except Exception as e:
        log.warning(f"[import] fail: {module_name} → {e}")
        return None

# Charts
save_burndown_chart = save_burnup_chart = save_cfd_chart = None
save_effort_vs_tasks_bubble = save_member_radar = None
save_member_radars_batch = save_workload_heatmap = save_skill_heatmap = None
save_eta_fan_chart = save_lorenz_workload = save_skill_concentration_bar = None
save_velocity_control_chart = save_cpm_gantt = save_sankey_effort = None
_build_pdf = None
charts_loaded = pdf_loaded = False

def _bind_charts_from(mod: types.ModuleType) -> bool:
    global save_member_radar_grid, save_skill_timeline_area, save_hhi_timeseries, save_bus_factor_bar
    save_member_radar_grid     = getattr(mod, "save_member_radar_grid", None)
    save_skill_timeline_area   = getattr(mod, "save_skill_timeline_area", None)
    save_hhi_timeseries        = getattr(mod, "save_hhi_timeseries", None)
    save_bus_factor_bar        = getattr(mod, "save_bus_factor_bar", None)

    global save_burndown_chart, save_burnup_chart, save_cfd_chart
    global save_effort_vs_tasks_bubble, save_member_radar, save_member_radars_batch
    global save_workload_heatmap, save_skill_heatmap, save_eta_fan_chart
    global save_lorenz_workload, save_skill_concentration_bar
    global save_velocity_control_chart, save_cpm_gantt, save_sankey_effort

    save_burndown_chart          = getattr(mod, "save_burndown_chart", getattr(mod, "save_burndown", None))
    save_burnup_chart            = getattr(mod, "save_burnup_chart",   getattr(mod, "save_burnup",   None))
    save_cfd_chart               = getattr(mod, "save_cfd_chart",      getattr(mod, "save_cfd",      None))
    save_effort_vs_tasks_bubble  = getattr(mod, "save_effort_vs_tasks_bubble", getattr(mod, "save_effort_vs_tasks", None))
    save_member_radar            = getattr(mod, "save_member_radar",   None)
    save_member_radars_batch     = getattr(mod, "save_member_radars_batch", None)
    save_workload_heatmap        = getattr(mod, "save_workload_heatmap", None)
    save_skill_heatmap           = getattr(mod, "save_skill_heatmap",    None)
    save_eta_fan_chart           = getattr(mod, "save_eta_fan_chart",    getattr(mod, "save_fan_chart", None))
    save_lorenz_workload         = getattr(mod, "save_lorenz_workload",  None)
    save_skill_concentration_bar = getattr(mod, "save_skill_concentration_bar", None)
    save_velocity_control_chart  = getattr(mod, "save_velocity_control_chart", None)
    save_cpm_gantt               = getattr(mod, "save_cpm_gantt", None)
    save_sankey_effort           = getattr(mod, "save_sankey_effort", None)
    return all([save_burndown_chart, save_burnup_chart, save_cfd_chart])

for name in ("proof_of_process.viz.charts_enterprise",
             "proof_of_process.viz.charts"):
    m = _try_import(name)
    if m and _bind_charts_from(m):
        charts_loaded = True
        break

# PDF
for name in ("proof_of_process.viz.pdf_report_enterprise",
             "proof_of_process.viz.pdf_report"):
    m = _try_import(name)
    if m and hasattr(m, "build_pdf"):
        _build_pdf = getattr(m, "build_pdf")
        pdf_loaded = True
        break

# Feature modules (optional)
mod_met = _try_import("proof_of_process.metrics.inequality")
mod_graph = _try_import("proof_of_process.graph.team_task")

mod_es   = _try_import("proof_of_process.evm.earned_schedule")
mod_cc   = _try_import("proof_of_process.anomaly.control_charts")
mod_cpm  = _try_import("proof_of_process.scheduling.cpm_pert")
mod_pri  = _try_import("proof_of_process.prioritization.scoring")
mod_asg  = _try_import("proof_of_process.planning.assignment")
mod_sim  = _try_import("proof_of_process.sim.what_if")

# ============ Helpers ============
def _ensure_dirs(outdir: str) -> Tuple[str, str, str]:
    outdir = outdir or "reports"
    figdir = os.path.join(outdir, "fig")
    expdir = os.path.join(outdir, "exports")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(expdir, exist_ok=True)
    return outdir, figdir, expdir

def _norm_text(s: str) -> str:
    import unicodedata, re
    s = str(s or "").strip()
    s2 = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").lower()
    return re.sub(r"\s+"," ", s2)

def _jw(a: str, b: str) -> float:
    a, b = _norm_text(a), _norm_text(b)
    if not a or not b: return 0.0
    la, lb = len(a), len(b)
    md = max(0, max(la, lb)//2 - 1)
    match, ha, hb = 0, [False]*la, [False]*lb
    for i in range(la):
        st, ed = max(0, i-md), min(i+md+1, lb)
        for j in range(st, ed):
            if hb[j]: continue
            if a[i]==b[j]: ha[i]=hb[j]=True; match += 1; break
    if not match: return 0.0
    t, p = 0, 0
    for i in range(la):
        if not ha[i]: continue
        while not hb[p]: p += 1
        if a[i]!=b[p]: t += 1
        p += 1
    t //= 2
    jaro = (match/la + match/lb + (match - t)/match)/3.0
    pref = 0
    for i in range(min(4, la, lb)):
        if a[i]==b[i]: pref += 1
        else: break
    return jaro + 0.1*pref*(1-jaro)

def _split_assignees(s: str) -> List[str]:
    import re
    s = str(s or "").strip()
    if not s: return []
    toks = re.split(r"(?:\+|[;,/|&])+", s)
    return [t.strip() for t in toks if t.strip()]

def _parse_json_list(val) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val or "").strip()
    if not s: return []
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list): return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            return []
    return _split_assignees(s)

def _coerce_week_index(df: pd.DataFrame, week_col: str = "Week") -> pd.DataFrame:
    out = df.copy()
    if week_col not in out.columns:
        out[week_col] = range(1, len(out) + 1)
        return out
    w = pd.to_numeric(out[week_col], errors="coerce")
    if w.isna().all() or (w.fillna(0).astype(int) <= 0).all():
        out[week_col] = range(1, len(out) + 1)
    else:
        out[week_col] = w.fillna(method="ffill").fillna(1).astype(int).clip(lower=1)
    return out

def _expand_by_members(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        members = _parse_json_list(r.get("Assignees_List","")) or _parse_json_list(r.get("Assigned_Member",""))
        if not members:
            tl = str(r.get("Task_Leader","")).strip()
            members = [tl] if tl else []
        for m in members:
            rr = r.to_dict()
            rr["Assigned_Member"] = str(m).strip()
            rows.append(rr)
    return pd.DataFrame(rows)

def _bucket_map_exec() -> Dict[str, List[str]]:
    return {
        "Frontend & Design": ["Frontend", "Design", "UI/UX"],
        "Backend": ["Backend"],
        "DevOps & Security": ["DevOps", "Security"],
        "Data & AI": ["Data/AI"],
        "QA & Testing": ["Testing/QA", "QA"],
        "Mobile": ["Mobile"],
        "PM & Strategy": ["Product/PM", "PM", "Strategy", "Biz", "Growth", "Content"],
        "Research & Insights": ["Research", "Market Research"],
        "General": ["General", "Documentation", "Support"],
    }

def _member_skill_profiles(df: pd.DataFrame, bucket_map: Dict[str, List[str]]) -> Dict[str, pd.Series]:
    import json as _json
    profiles = {}
    def _skills(v):
        if isinstance(v, list): return v
        s = str(v or "")
        if s.strip().startswith("["):
            try: return _json.loads(s)
            except: return []
        return []
    all_buckets = list(bucket_map.keys())
    inv = {}
    for b, cats in bucket_map.items():
        for c in cats:
            inv[_norm_text(c)] = b
    for m, grp in df.groupby("Assigned_Member", dropna=True):
        agg = {b:0.0 for b in all_buckets}
        for _, r in grp.iterrows():
            eff = float(pd.to_numeric(r.get("Effort_Score", 1.0), errors="coerce") or 1.0)
            for sk in _skills(r.get("Skills", [])):
                b = inv.get(_norm_text(sk), "General" if "General" in agg else all_buckets[-1])
                agg[b] += eff
        vec = pd.Series(agg, index=all_buckets)
        mx = float(vec.max()) if vec.max() > 0 else 1.0
        profiles[str(m)] = (vec / mx * 100.0).round(2)
    return profiles

def _gini(series: pd.Series) -> float:
    x = series.dropna().astype(float).clip(lower=0).values
    if x.size == 0: return 0.0
    if np.allclose(x.sum(), 0): return 0.0
    x = np.sort(x)
    n = len(x); cumx = np.cumsum(x)
    B = cumx.sum() / (x.sum() * n)
    return 1 + (1/n) - 2*B

def _theil(series: pd.Series) -> float:
    x = series.dropna().astype(float).clip(lower=0).values
    if x.size == 0: return 0.0
    mu = x.mean()
    if mu <= 0: return 0.0
    x = x[x>0]
    return float(np.mean((x/mu) * np.log(x/mu)))

def _compute_burn_series(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    d = data.copy()
    d["Effort_Score"] = pd.to_numeric(d["Effort_Score"], errors="coerce").fillna(1.0)
    done = d["Completion_Status"].str.contains("Completed", case=False, na=False)

    scope = d.groupby("Week")["Effort_Score"].sum().cumsum().rename("ScopeCumulative")
    completed = d.loc[done].groupby("Week")["Effort_Score"].sum().reindex(scope.index, fill_value=0.0).cumsum()
    burnup = pd.concat([scope, completed.rename("CompletedCumulative")], axis=1).reset_index()

    burndown = burnup.copy()
    burndown["RemainingEffort"] = burndown["ScopeCumulative"] - burndown["CompletedCumulative"]
    burndown = burndown[["Week","RemainingEffort"]].copy()
    burndown["Week"] = pd.to_numeric(burndown["Week"], errors="coerce").fillna(1).astype(int)

    d["Completion_Status"] = d["Completion_Status"].astype(str).str.title()
    piv = (d.pivot_table(index="Week", columns="Completion_Status",
                         values="Effort_Score", aggfunc="sum", fill_value=0.0)
             .reindex(columns=["Not Started","In Progress","Completed"], fill_value=0.0))
    cfd = piv.cumsum().reset_index()
    cfd["Week"] = pd.to_numeric(cfd["Week"], errors="coerce").fillna(1).astype(int)
    return {"burndown": burndown, "burnup": burnup, "cfd": cfd}

def _charts_smoke():
    fns = [
        ("save_burndown_chart", save_burndown_chart),
        ("save_burnup_chart", save_burnup_chart),
        ("save_cfd_chart", save_cfd_chart),
        ("save_member_radar", save_member_radar),
        ("save_skill_heatmap", save_skill_heatmap),
        ("save_workload_heatmap", save_workload_heatmap),
        ("save_velocity_control_chart", save_velocity_control_chart),
    ]
    return True, [name for name, fn in fns if callable(fn)]

# ============ Main pipeline ============
def run_pipeline(df: pd.DataFrame, outdir: str = "reports") -> Dict[str, Any]:
    warnings.filterwarnings("ignore")
    outdir, figdir, expdir = _ensure_dirs(outdir)

    # 0) Chuẩn hoá nhẹ
    need_cols = [
        "Project","Task","Assigned_Member","Task_Leader","Assignees_List","Skills",
        "Effort_Score","Planned_Effort","Actual_Cost","Week","Completion_Status","Depends_On",
        "Business_Value","Time_Criticality","Risk_Reduction","Job_Size","Reach","Impact","Confidence","Effort"
    ]
    data = df.copy()
    for c in need_cols:
        if c not in data.columns:
            data[c] = "" if c not in ["Effort_Score","Planned_Effort","Actual_Cost","Week",
                                      "Business_Value","Time_Criticality","Risk_Reduction","Job_Size",
                                      "Reach","Impact","Confidence","Effort"] else np.nan
    data = _coerce_week_index(data, "Week")
    for c in ["Effort_Score","Planned_Effort","Actual_Cost","Business_Value","Time_Criticality",
              "Risk_Reduction","Job_Size","Reach","Impact","Confidence","Effort"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # 1) Burn series
    burn = _compute_burn_series(data)
    burndown, burnup, cfd = burn["burndown"], burn["burnup"], burn["cfd"]

    # 2) Velocity series & Control charts (EWMA/CUSUM)
    fig_paths: List[str] = []
    velocity = burnup["CompletedCumulative"].diff().fillna(burnup["CompletedCumulative"]).clip(lower=0)
    vel_chart = None
    ewma_signals = {}
    if mod_cc and callable(getattr(mod_cc, "control_signals", None)):
        try:
            ewma_signals = mod_cc.control_signals(velocity)
            if charts_loaded and callable(save_velocity_control_chart):
                vel_chart = save_velocity_control_chart(velocity, ewma_signals,
                                                        os.path.join(figdir, "velocity_control.png"))
                if vel_chart: fig_paths.append(vel_chart)
        except Exception as e:
            log.warning(f"Control charts failed: {e}")

    # 3) ETA (Monte Carlo) — giữ như trước nếu bạn có hàm
    weeks_dist = None
    eta = {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    remaining = float(data.loc[~data["Completion_Status"].str.contains("Completed", case=False, na=False), "Effort_Score"].sum())
    try:
        # velocity as array
        vel = velocity.values
        if np.nansum(vel) > 0:
            # simple MC if module 'forecasting' của bạn chưa có
            samples = max(2000, 500)
            vel_pos = np.where(vel<=0, np.nan, vel).astype(float)
            mu = np.nanmean(vel_pos); sd = np.nanstd(vel_pos)
            sd = sd if sd>0 else (mu*0.25 if mu>0 else 1.0)
            draws = np.clip(np.random.normal(mu, sd, size=samples), 1e-6, None)
            weeks_dist = np.ceil(remaining / draws)
            eta = {"p10": float(np.quantile(weeks_dist, 0.10)),
                   "p50": float(np.quantile(weeks_dist, 0.50)),
                   "p90": float(np.quantile(weeks_dist, 0.90))}
            if charts_loaded and callable(save_eta_fan_chart):
                p = save_eta_fan_chart(weeks_dist, os.path.join(figdir, "eta_fan.png"))
                if p: fig_paths.append(p)
    except Exception as e:
        log.warning(f"MC ETA failed: {e}")

    # 4) Earned Schedule & EVM time-based
    es_metrics = {}
    if mod_es and callable(getattr(mod_es, "earned_schedule", None)):
        try:
            es_metrics = mod_es.earned_schedule(burnup)
        except Exception as e:
            log.warning(f"Earned Schedule failed: {e}")

    # 5) Mở rộng theo member & hồ sơ kỹ năng
    expanded = _expand_by_members(data)
    bucket_map = _bucket_map_exec()
    member_profiles = _member_skill_profiles(expanded, bucket_map)

    # 6) Charts cốt lõi (burn/cfd/heatmap/radar/lorenz/hhi/sankey…)
    if charts_loaded:
        try:
            if not burndown.empty: fig_paths.append(save_burndown_chart(burndown, os.path.join(figdir, "burndown.png")))
            if not burnup.empty:   fig_paths.append(save_burnup_chart(burnup, os.path.join(figdir, "burnup.png")))
            if not cfd.empty:      fig_paths.append(save_cfd_chart(cfd, os.path.join(figdir, "cfd.png")))
            if "Assigned_Member" in expanded.columns and not expanded.empty and callable(save_workload_heatmap):
                fig_paths.append(save_workload_heatmap(expanded, os.path.join(figdir, "workload_heatmap.png")))
            if "Skills" in expanded.columns and not expanded.empty and callable(save_skill_heatmap):
                fig_paths.append(save_skill_heatmap(expanded, os.path.join(figdir, "skill_heatmap.png")))
            # Radar — render top 8 theo effort
            if member_profiles and callable(save_member_radar):
                mdir = os.path.join(figdir, "members"); os.makedirs(mdir, exist_ok=True)
                eff_by_member = expanded.groupby("Assigned_Member")["Effort_Score"].sum().sort_values(ascending=False)
                for m in eff_by_member.index[:8]:
                    if m in member_profiles:
                        safe = "".join([c for c in m if c.isalnum() or c in "-_ ."]).strip().replace(" ", "_")
                        p = os.path.join(mdir, f"radar_{safe}.png")
                        try:
                            save_member_radar(member_profiles[m], m, p)
                            fig_paths.append(p)
                        except Exception as e:
                            log.warning(f"Radar failed for {m}: {e}")
            # radar grid (nhanh để so sánh nhiều người)
            if member_profiles and callable(save_member_radar_grid):
                p = save_member_radar_grid(member_profiles, os.path.join(figdir, "radar_grid.png"), limit=9)
                if p: fig_paths.append(p)

            # skill timeline + HHI time-series
            if callable(save_skill_timeline_area):
                p = save_skill_timeline_area(expanded, os.path.join(figdir, "skill_timeline.png"))
                if p: fig_paths.append(p)
            if callable(save_hhi_timeseries):
                p = save_hhi_timeseries(expanded, os.path.join(figdir, "hhi_timeseries.png"))
                if p: fig_paths.append(p)

            # Lorenz & HHI
            effort_per_member = expanded.groupby("Assigned_Member")["Effort_Score"].sum()
            if not effort_per_member.empty and callable(save_lorenz_workload):
                fig_paths.append(save_lorenz_workload(effort_per_member, os.path.join(figdir, "lorenz_workload.png")))
            if member_profiles and callable(save_skill_concentration_bar):
                fig_paths.append(save_skill_concentration_bar(member_profiles, os.path.join(figdir, "skill_hhi.png")))
            # Sankey effort (skills -> member)
            if callable(save_sankey_effort) and "Skills" in expanded.columns:
                sankey = save_sankey_effort(expanded, os.path.join(figdir, "sankey_effort.png"))
                if sankey: fig_paths.append(sankey)
        except Exception as e:
            log.warning(f"Charts failed: {e}")
    else:
        log.warning("Charts module not found; skipping figures.")
    # 7) CPM/PERT + mô phỏng
    # 7) CPM/PERT + mô phỏng
    cpm_summary = {}
    if mod_cpm:
        try:
            cpm_df, csum = mod_cpm.compute_cpm(data)
        except Exception as e:
            log.warning(f"CPM strict failed: {e}; falling back to relaxed CPM (SCC-condensed).")
            cpm_df, csum = mod_cpm.compute_cpm_relaxed(data)

        try:
            cpm_summary = csum or {}
            cpm_df.to_csv(os.path.join(expdir, "07_cpm_table.csv"), index=False, encoding="utf-8-sig")
            # export chu trình nếu có
            cyc = csum.get("cycles") if isinstance(csum, dict) else None
            if cyc:
                import csv
                with open(os.path.join(expdir, '07_cpm_cycles.csv'), 'w', newline='', encoding='utf-8-sig') as f:
                    w = csv.writer(f); w.writerow(["Cycle_Group"])
                    for group in cyc: w.writerow([" -> ".join(group)])

            if charts_loaded and callable(save_cpm_gantt) and not cpm_df.empty:
                p = save_cpm_gantt(cpm_df, os.path.join(figdir, "cpm_gantt.png"))
                if p: fig_paths.append(p)

            # PERT Monte Carlo
            pert = mod_cpm.monte_carlo_pert(data, samples=2000)
            if pert and "duration_samples" in pert:
                np.save(os.path.join(expdir, "08_pert_duration.npy"), pert["duration_samples"])
        except Exception as e:
            log.warning(f"CPM/PERT failed: {e}")


    # 7.1) Bus-factor / key-person risk
    bus_df = pd.DataFrame()
    if mod_graph:
        try:
            bus_df = mod_graph.metrics_bus_factor(expanded)
            bus_df.to_csv(os.path.join(expdir, "13_bus_factor.csv"), index=False, encoding="utf-8-sig")
            if charts_loaded and callable(save_bus_factor_bar) and not bus_df.empty:
                p = save_bus_factor_bar(bus_df, os.path.join(figdir, "bus_factor.png"))
                if p: fig_paths.append(p)
        except Exception as e:
            log.warning(f"Bus-factor analysis failed: {e}")

    # 8) Ưu tiên: WSJF & RICE
    pri_tables = {}
    if mod_pri:
        try:
            wsjf = mod_pri.compute_wsjf(data.copy())
            rice = mod_pri.compute_rice(data.copy())
            wsjf.to_csv(os.path.join(expdir, "09_wsjf.csv"), index=False, encoding="utf-8-sig")
            rice.to_csv(os.path.join(expdir, "10_rice.csv"), index=False, encoding="utf-8-sig")
            pri_tables = {"wsjf": wsjf.head(20), "rice": rice.head(20)}
        except Exception as e:
            log.warning(f"Prioritization failed: {e}")

    # 9) Gán việc (assignment) — gợi ý
    if mod_asg and member_profiles:
        try:
            suggest = mod_asg.suggest_assignments(data.copy(), member_profiles, topn=50)
            suggest.to_csv(os.path.join(expdir, "11_assignment_suggestions.csv"), index=False, encoding="utf-8-sig")
        except Exception as e:
            log.warning(f"Assignment suggest failed: {e}")

    # 10) What-if (mặc định chạy 2 preset nhẹ)
    if mod_sim:
        try:
            what = mod_sim.run_scenarios(remaining_effort=remaining, velocity=velocity.values)
            with open(os.path.join(expdir, "12_what_if.json"), "w", encoding="utf-8") as f:
                json.dump(what, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"What-if failed: {e}")

    # 11) Summary & metrics
    all_members = sorted(set(expanded["Assigned_Member"].dropna().astype(str).tolist()))
    effort_per_member = expanded.groupby("Assigned_Member")["Effort_Score"].sum()
    gini = float(_gini(effort_per_member)) if not effort_per_member.empty else 0.0
    theil = float(_theil(effort_per_member)) if not effort_per_member.empty else 0.0
    avg_hhi = float(np.mean([

        float(np.sum((vec/(vec.sum() if vec.sum()>0 else 1))**2)) for vec in member_profiles.values()
    ])) if member_profiles else 0.0

    summary = {
        "Projects": int(data["Project"].nunique()) if "Project" in data.columns else 0,
        "Members": int(len(all_members)),
        "Overall Effort (sum)": float(pd.to_numeric(data["Effort_Score"], errors="coerce").fillna(0).sum()),
        "Remaining Effort": remaining,
        "ETA weeks (P50)": float(eta.get("p50", 0.0) or 0.0),
        "Risk": "Medium (score 50)",
        "Workload Inequality (Gini)": round(gini, 3),
        "Workload Inequality (Theil)": round(theil, 3),
        "Avg Skill Concentration (HHI)": round(avg_hhi, 3),
        # ES time metrics (nếu có)
        "SPI(t)": round(float(es_metrics.get("SPI_t", np.nan)), 3) if es_metrics else None,
        "SV(t)": round(float(es_metrics.get("SV_t", np.nan)), 3) if es_metrics else None,
        "Critical Path Duration": float(cpm_summary.get("duration", 0.0)) if cpm_summary else 0.0,
    }
    # inequality advanced
    if mod_met:
        try:
            at = mod_met.atkinson(effort_per_member, eps=0.5)
            hv = mod_met.hoover(effort_per_member)
        except Exception:
            at = hv = np.nan
    else:
        at = hv = np.nan
    summary.update({
        "Atkinson (ε=0.5)": round(float(at), 3) if np.isfinite(at) else None,
        "Hoover": round(float(hv), 3) if np.isfinite(hv) else None,
    })


    # 12) Xuất CSV
    try:
        data.to_csv(os.path.join(expdir, "01_cleaned_enriched.csv"), index=False, encoding="utf-8-sig")
        burndown.to_csv(os.path.join(expdir, "03_burndown.csv"), index=False, encoding="utf-8-sig")
        burnup.to_csv(os.path.join(expdir, "04_burnup.csv"), index=False, encoding="utf-8-sig")
        cfd.to_csv(os.path.join(expdir, "05_cfd.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(member_profiles).T.to_csv(os.path.join(expdir, "06_member_skill_profiles.csv"),
                                               encoding="utf-8-sig")
    except Exception as e:
        log.warning(f"Export CSV failed: {e}")

    # 13) Build PDF
    pdf_path = os.path.join(outdir, "Executive_Report.pdf")
    try:
        if _build_pdf is not None:
            _build_pdf(
                title="Proof of Process — Executive Report",
                summary_df=pd.DataFrame([summary]),
                charts=[p for p in (fig_paths or []) if p],
                tables={k: v for k, v in pri_tables.items() if isinstance(v, pd.DataFrame)},
                out_path=pdf_path
            )
        else:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            styles = getSampleStyleSheet()
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            story = [Paragraph("Proof of Process — Executive Report", styles["Title"]), Spacer(1,12)]
            for k,v in summary.items():
                story.append(Paragraph(f"<b>{k}</b>: {v}", styles["Normal"]))
                story.append(Spacer(1,6))
            doc.build(story)
        log.info(f"Report saved: {pdf_path}")
    except Exception as e:
        log.warning(f"PDF build failed: {e}")

    return {
        "summary": summary,
        "pdf": pdf_path,
        "exports_dir": expdir,
        "fig_dir": figdir,
    }
