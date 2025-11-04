# proof_of_process/viz/charts_enterprise.py
from __future__ import annotations
import os, math, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ====== THEME (Executive Dark) ======
PALETTE = {
    "bg":      "#0B1021",
    "panel":   "#12172B",
    "grid":    "#2B3150",
    "text":    "#EAF0FF",
    "muted":   "#A7B0C8",
    "primary": "#5B8FF9",
    "accent":  "#5AD8A6",
    "warn":    "#F6BD16",
    "danger":  "#F4664A",
    "purple":  "#9B5DE5",
    "ink":     "#0E122B",
}

def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["panel"],
        "savefig.facecolor":PALETTE["bg"],
        "axes.edgecolor":   PALETTE["grid"],
        "axes.labelcolor":  PALETTE["text"],
        "axes.titlecolor":  PALETTE["text"],
        "xtick.color":      PALETTE["muted"],
        "ytick.color":      PALETTE["muted"],
        "grid.color":       PALETTE["grid"],
        "text.color":       PALETTE["text"],
        "font.size":        11,
        "axes.titleweight": "bold",
        "axes.grid":        True,
        "grid.linestyle":   "--",
        "grid.linewidth":   0.6,
        "legend.frameon":   False,
    })

def _save(figpath: str):
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath, dpi=220, bbox_inches="tight")
    plt.close()
    return figpath

def _annotate_last(x, y, label_fmt="{:.1f}", color=None):
    if len(x) == 0:
        return
    xv, yv = x[-1], y[-1]
    s = label_fmt.format(yv) if isinstance(label_fmt, str) else str(label_fmt(yv))
    plt.scatter([xv], [yv], s=40, color=color or PALETTE["accent"], zorder=5)
    plt.annotate(s, (xv, yv), textcoords="offset points", xytext=(8, 6),
                 color=color or PALETTE["accent"])

# ====== CORE CHARTS (giữ nguyên API cũ) ======
def save_burndown_chart(df: pd.DataFrame, out_path: str):
    if df is None or df.empty:
        return None
    _apply_theme()
    plt.figure(figsize=(8.5, 5.2))
    x = df["Week"].astype(float).values
    y = df["RemainingEffort"].astype(float).values
    plt.plot(x, y, marker="o", linewidth=2.2, color=PALETTE["primary"])
    _annotate_last(x, y, color=PALETTE["primary"])
    plt.title("Burndown — Remaining Effort by Week")
    plt.xlabel("Week"); plt.ylabel("Remaining Effort")
    return _save(out_path)

def save_burnup_chart(df: pd.DataFrame, out_path: str):
    if df is None or df.empty:
        return None
    _apply_theme()
    plt.figure(figsize=(8.5, 5.2))
    x = df["Week"].astype(float).values
    if "ScopeCumulative" in df.columns:
        y1 = df["ScopeCumulative"].astype(float).values
        plt.plot(x, y1, marker="o", linewidth=2.0, color=PALETTE["muted"], label="Scope")
        _annotate_last(x, y1, color=PALETTE["muted"])
    if "CompletedCumulative" in df.columns:
        y2 = df["CompletedCumulative"].astype(float).values
        plt.plot(x, y2, marker="o", linewidth=2.2, color=PALETTE["accent"], label="Completed")
        _annotate_last(x, y2, color=PALETTE["accent"])
    plt.legend()
    plt.title("Burnup — Scope vs Completed")
    plt.xlabel("Week"); plt.ylabel("Cumulative Effort")
    return _save(out_path)

def save_cfd_chart(df: pd.DataFrame, out_path: str):
    if df is None or df.empty:
        return None
    _apply_theme()
    plt.figure(figsize=(8.5, 5.2))
    w  = df["Week"].astype(float).values
    ns = df.get("Not Started", pd.Series([0]*len(w))).astype(float).values
    ip = df.get("In Progress", pd.Series([0]*len(w))).astype(float).values
    dn = df.get("Completed",   pd.Series([0]*len(w))).astype(float).values
    plt.stackplot(w, ns, ip, dn,
                  colors=[PALETTE["muted"], PALETTE["primary"], PALETTE["accent"]],
                  labels=["Not Started", "In Progress", "Completed"], alpha=0.95)
    plt.legend(loc="upper left")
    plt.title("Cumulative Flow Diagram")
    plt.xlabel("Week"); plt.ylabel("Cumulative Effort")
    return _save(out_path)

def save_effort_vs_tasks_bubble(df: pd.DataFrame, out_path: str):
    if df is None or df.empty or "Assigned_Member" not in df.columns:
        return None
    _apply_theme()
    agg = (df.groupby("Assigned_Member")
             .agg(Tasks=("Task","count"),
                  Effort=("Effort_Score","sum"))
             .reset_index())
    plt.figure(figsize=(8.5, 5.2))
    cmap = mpl.colormaps.get("tab10")
    for i, r in agg.iterrows():
        s = math.sqrt(max(1.0, r["Effort"])) * 18
        plt.scatter(r["Tasks"], r["Effort"], s=s, color=cmap(i%10), alpha=0.9, label=r["Assigned_Member"])
        plt.text(r["Tasks"]+0.05, r["Effort"]+0.3, r["Assigned_Member"], color=PALETTE["text"])
    plt.xlabel("Total Tasks")
    plt.ylabel("Total Effort")
    plt.title("Performance Bubble — Tasks vs Effort")
    plt.legend([], frameon=False)
    return _save(out_path)

def save_member_radar(scores: pd.Series, member_name: str, out_path: str):
    """
    scores: index = skill buckets, values = 0..100
    """
    if scores is None or scores.empty:
        return None
    vals = scores.values.astype(float)
    labs = list(scores.index)
    vals = np.append(vals, vals[0])                     # khép vòng
    angles = np.linspace(0, 2*np.pi, len(labs), endpoint=False)
    angles = np.append(angles, angles[0])
    _apply_theme()
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(PALETTE["panel"])
    ax.plot(angles, vals, color=PALETTE["primary"], linewidth=2.0)
    ax.fill(angles, vals, color=PALETTE["primary"], alpha=0.25)
    ax.set_thetagrids(angles[:-1]*180/np.pi, labs, frac=1.15)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)
    ax.grid(True)
    plt.title(f"Biểu đồ Năng lực: {member_name}", loc="left")
    return _save(out_path)

def save_workload_heatmap(df: pd.DataFrame, out_path: str):
    """
    Heatmap Workload: hàng=Member, cột=Week, giá trị=sum Effort.
    """
    if df is None or df.empty:
        return None
    _apply_theme()
    piv = (df.groupby(["Assigned_Member","Week"])["Effort_Score"]
             .sum().unstack(fill_value=0.0))
    if piv.empty: return None
    plt.figure(figsize=(9.5, 5.8))
    im = plt.imshow(piv.values, aspect="auto", cmap=mpl.colormaps.get("magma"))
    plt.colorbar(im, label="Effort Sum")
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=0)
    plt.title("Workload Heatmap — Effort per Week per Member")
    return _save(out_path)

def save_skill_heatmap(data: pd.DataFrame, out_path: str):
    """
    Heatmap kỹ năng: Member x Skill, tổng effort
    """
    if data is None or data.empty or "Skills" not in data.columns:
        return None
    def _skills(v):
        if isinstance(v, list): return v
        if isinstance(v, str) and v.strip().startswith("["):
            try: return json.loads(v)
            except: return []
        return []
    rows = []
    for _, r in data.iterrows():
        m = str(r.get("Assigned_Member", "")).strip()
        if not m: continue
        for sk in _skills(r["Skills"]):
            rows.append((m, str(sk), float(r.get("Effort_Score", 1.0))))
    if not rows:
        return None
    mat = (pd.DataFrame(rows, columns=["Member","Skill","Effort"])
           .groupby(["Member","Skill"])["Effort"].sum()
           .unstack(fill_value=0.0))
    _apply_theme()
    plt.figure(figsize=(8.5, 5.2))
    im = plt.imshow(mat.values, aspect="auto", cmap=mpl.colormaps.get("viridis"))
    plt.colorbar(im, label="Effort Sum")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    plt.title("Skill Heatmap")
    return _save(out_path)

def save_eta_fan_chart(weeks_dist: np.ndarray, out_path: str):
    arr = np.asarray(weeks_dist, dtype=float)
    if arr.size == 0:
        return None
    _apply_theme()
    plt.figure(figsize=(8.5, 5.2))
    bins = min(30, max(6, int(np.sqrt(arr.size))))
    plt.hist(arr, bins=bins, density=True, color=PALETTE["warn"], alpha=0.95)
    p10, p50, p90 = np.quantile(arr, [0.1, 0.5, 0.9])
    for v, name, col in [(p10, "P10", PALETTE["muted"]),
                         (p50, "P50", PALETTE["accent"]),
                         (p90, "P90", PALETTE["danger"])]:
        plt.axvline(v, linestyle="--", color=col, linewidth=2.0, label=f"{name}={v:.1f}")
    plt.legend()
    plt.title("ETA Fan (Monte Carlo)")
    plt.xlabel("Weeks to Complete"); plt.ylabel("Density")
    return _save(out_path)

# ====== NEW: ECON / PM ANALYTICS CHARTS ======
def save_lorenz_workload(series: pd.Series, out_path: str, title: str = "Lorenz Curve — Workload Equality"):
    """
    Vẽ đường Lorenz cho phân phối workload (effort theo member) + hiển thị hệ số Gini & Theil.
    """
    s = series.dropna().astype(float).clip(lower=0)
    if s.empty: return None
    s_sorted = s.sort_values()
    cum = s_sorted.cumsum()
    cum_share = cum / cum.iloc[-1]
    n = len(s_sorted)
    x = np.linspace(0, 1, n+1)
    y = np.concatenate([[0.0], cum_share.values])

    # Equality line
    _apply_theme()
    plt.figure(figsize=(7.8, 5.2))
    plt.plot([0,1],[0,1], linestyle="--", color=PALETTE["muted"], label="Perfect Equality")
    plt.plot(x, y, color=PALETTE["primary"], linewidth=2.2, label="Lorenz")

    # Gini
    # 2*area between equality and Lorenz
    B = np.trapz(y, x)
    gini = 1 - 2*B

    # Theil (T) = mean( (xi/μ)*ln(xi/μ) )
    xvals = s_sorted.values
    mu = xvals.mean()
    safe = xvals[xvals>0]
    theil = float(np.mean((safe/mu)*np.log(safe/mu))) if mu>0 else np.nan

    plt.text(0.6, 0.2, f"Gini={gini:.3f}\nTheil={theil:.3f}", transform=plt.gca().transAxes,
             color=PALETTE["text"], bbox=dict(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"]))
    plt.title(title); plt.xlabel("Cumulative share of Members"); plt.ylabel("Cumulative share of Effort")
    plt.legend(loc="lower right")
    return _save(out_path)

def save_skill_concentration_bar(profiles: dict[str, pd.Series], out_path: str, topn: int = 20):
    """
    Thanh HHI (Herfindahl–Hirschman) đo mức "tập trung kỹ năng" của từng member.
    HHI = sum(p_i^2), với p_i là tỷ trọng effort theo bucket kỹ năng.
    """
    if not profiles:
        return None
    scores = {}
    for m, s in profiles.items():
        p = (s / (s.sum() if s.sum()>0 else 1)).clip(lower=0)
        scores[m] = float(np.sum(np.square(p.values)))
    ser = pd.Series(scores).sort_values(ascending=False).head(topn)

    _apply_theme()
    plt.figure(figsize=(9.2, 5.2))
    bars = plt.bar(ser.index, ser.values, color=PALETTE["purple"])
    plt.xticks(rotation=30, ha="right")
    for b in bars:
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.2f}",
                 ha="center", va="bottom", color=PALETTE["text"])
    plt.title("Skill Concentration (HHI) — Higher = Narrower Skill Portfolio")
    plt.ylabel("HHI (0..1)")
    return _save(out_path)

# ====== NEW: RADAR BATCH ======
# thêm ở đầu file (gần import) nếu chưa có:
import matplotlib.patheffects as pe
import numpy as np
import os
import matplotlib.pyplot as plt
# (giữ nguyên PALETTE và _apply_theme() của bạn)

# ---------- helper: nền radar với các vòng mờ xen kẽ ----------
def _radar_background(ax, rings=(20, 40, 60, 80, 100)):
    """
    Tô nền theo dải đồng tâm mờ để dễ đọc. Không phụ thuộc version matplotlib.
    """
    # vẽ các dải nền xen kẽ
    theta = np.linspace(0, 2*np.pi, 360)
    last_r = 0.0
    for i, r in enumerate(rings):
        color = PALETTE["panel"] if i % 2 == 0 else PALETTE["bg"]
        ax.fill_between(theta, last_r, r, color=color, alpha=0.35, zorder=0)
        last_r = r

    # định dạng lưới/ticks
    ax.set_ylim(0, rings[-1])
    ax.set_yticks(rings)
    ax.set_yticklabels([str(r) for r in rings], color=PALETTE["muted"], fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color=PALETTE["grid"])
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color=PALETTE["grid"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
        spine.set_linewidth(1.0)

# ---------- hàm chính: radar đẹp hơn ----------
def save_member_radar(scores: pd.Series, member_name: str, out_path: str):
    """
    scores: index = các bucket kỹ năng, values = 0..100
    """
    if scores is None or scores.empty:
        return None

    # chuẩn hoá dữ liệu
    labels = list(scores.index)
    values = scores.values.astype(float)

    _apply_theme()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # hướng & gốc: bắt đầu ở 12h, quay theo chiều kim đồng hồ
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    # toạ độ góc cho từng trục
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    # nền + vòng đồng tâm
    _radar_background(ax, rings=(20, 40, 60, 80, 100))

    # vẽ đa giác
    angles_closed = np.concatenate([angles, [angles[0]]])
    values_closed = np.concatenate([values, [values[0]]])

    ax.plot(angles_closed, values_closed,
            color=PALETTE["primary"], linewidth=2.4, solid_capstyle="round")
    ax.fill(angles_closed, values_closed,
            color=PALETTE["primary"], alpha=0.28)

    # điểm nút + nhãn giá trị (ẩn giá trị nhỏ để tránh rối)
    ax.scatter(angles, values, s=38, zorder=3, color=PALETTE["accent"])
    for a, v in zip(angles, values):
        if v >= 12:  # ngưỡng hiển thị
            ax.text(a, v + 5, f"{v:.0f}",
                    ha="center", va="center", fontsize=9, color=PALETTE["text"],
                    path_effects=[pe.withStroke(linewidth=2, foreground=PALETTE["ink"])])

    # nhãn trục (xoay theo hướng trục, có outline để dễ đọc trên nền tối)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=12)
    for t, a in zip(ax.get_xticklabels(), angles):
        t.set_fontsize(11)
        t.set_rotation(np.degrees(a) - 90)
        t.set_rotation_mode("anchor")
        t.set_horizontalalignment("center")
        t.set_color(PALETTE["muted"])
        t.set_path_effects([pe.withStroke(linewidth=3, foreground=PALETTE["bg"])])

    # highlight kỹ năng mạnh nhất
    if np.max(values) > 0:
        k = int(np.argmax(values))
        ax.annotate(f"Top: {labels[k]} ({values[k]:.0f})",
                    xy=(angles[k], values[k]),
                    xytext=(0.5*np.pi, 0.0), textcoords="offset points",
                    fontsize=10, color=PALETTE["text"])

    # tiêu đề to, rõ
    title_txt = f"Biểu đồ Năng lực: {member_name}"
    fig.suptitle(title_txt, x=0.07, y=0.97, ha="left", va="top",
                 fontsize=20, fontweight="heavy",
                 path_effects=[pe.withStroke(linewidth=3, foreground=PALETTE["bg"])])

    # lưu
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_member_radars_batch(profiles: dict[str, pd.Series], out_dir: str) -> list[str]:
    """
    Lưu radar cho tất cả member trong dict {member -> Series(buckets 0..100)}.
    Trả về danh sách đường dẫn ảnh.
    """
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for name, vec in profiles.items():
        safe_name = "".join([c for c in name if c.isalnum() or c in "-_ ."]).strip().replace(" ", "_")
        path = os.path.join(out_dir, f"radar_{safe_name}.png")
        try:
            save_member_radar(vec, name, path)
            out.append(path)
        except Exception as e:
            # không dừng batch nếu 1 người lỗi
            print(f"[radar] skip {name}: {e}")
    return out

# --- ADDITIONS in charts_enterprise.py ---
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
import matplotlib as mpl

def save_velocity_control_chart(vel: pd.Series, sig: dict, out_path: str):
    if vel is None or len(vel)==0: return None
    _apply_theme()
    x = np.arange(1, len(vel)+1)
    y = np.asarray(vel, dtype=float)
    ew = np.asarray(sig.get("ewma", np.zeros_like(y)), dtype=float)
    UCL = np.asarray(sig.get("UCL", np.zeros_like(y)))
    LCL = np.asarray(sig.get("LCL", np.zeros_like(y)))

    plt.figure(figsize=(9,5.2))
    plt.plot(x, y, marker="o", linewidth=1.6, color=PALETTE["accent"], label="Velocity")
    plt.plot(x, ew, linewidth=2.2, color=PALETTE["primary"], label="EWMA")
    plt.plot(x, UCL, linestyle="--", color=PALETTE["danger"], label="UCL")
    plt.plot(x, LCL, linestyle="--", color=PALETTE["warn"], label="LCL")
    for i in sig.get("alarms_pos", []):
        plt.scatter([x[i]], [y[i]], s=60, color=PALETTE["danger"])
    for i in sig.get("alarms_neg", []):
        plt.scatter([x[i]], [y[i]], s=60, color=PALETTE["warn"])
    plt.legend()
    plt.title("Velocity Control Chart (EWMA/CUSUM)")
    plt.xlabel("Week"); plt.ylabel("Done Effort per Week")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    return out_path

def save_cpm_gantt(cpm_df: pd.DataFrame, out_path: str):
    if cpm_df is None or cpm_df.empty: return None
    _apply_theme()
    df = cpm_df.copy().sort_values("ES")
    names = df["Task"].tolist()
    starts = df["ES"].tolist()
    durations = df["Duration"].tolist()
    crit = df["Critical"].tolist()

    fig, ax = plt.subplots(figsize=(10, .5*len(df)+2))
    for i,(s,d,c) in enumerate(zip(starts, durations, crit)):
        ax.barh(i, d, left=s, color=PALETTE["accent"] if c else PALETTE["muted"], alpha=0.9, height=0.5)
        ax.text(s+d+0.1, i, f"{d:.1f}", va="center", color=PALETTE["text"])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Time (weeks)"); ax.set_title("CPM Gantt — ES/EF (Critical in green)")
    ax.invert_yaxis()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    return out_path

def save_sankey_effort(expanded_df: pd.DataFrame, out_path: str, top_skills: int = 6, top_members: int = 8):
    """
    Sơ đồ luồng effort từ Skill bucket -> Member (dựa trên expanded_df đã có cột Skills & Assigned_Member).
    Implement đơn giản bằng heatmap 'song song' để tránh phụ thuộc ngoài.
    """
    if expanded_df is None or expanded_df.empty or "Skills" not in expanded_df.columns:
        return None
    # rút gọn top kỹ năng và top thành viên
    def _skills(v):
        if isinstance(v, list): return v
        s = str(v or "")
        if s.strip().startswith("["):
            import json
            try: return [str(x) for x in json.loads(s)]
            except: return []
        return []
    rows=[]
    for _, r in expanded_df.iterrows():
        m = str(r.get("Assigned_Member","")).strip()
        if not m: continue
        eff = float(pd.to_numeric(r.get("Effort_Score", 1.0), errors="coerce") or 1.0)
        for sk in _skills(r["Skills"]):
            rows.append((sk, m, eff))
    if not rows: return None
    df = (pd.DataFrame(rows, columns=["Skill","Member","Effort"])
          .groupby(["Skill","Member"])["Effort"].sum().reset_index())
    top_sk = df.groupby("Skill")["Effort"].sum().sort_values(ascending=False).head(top_skills).index
    top_mb = df.groupby("Member")["Effort"].sum().sort_values(ascending=False).head(top_members).index
    df = df[df["Skill"].isin(top_sk) & df["Member"].isin(top_mb)]
    if df.empty: return None
    # tạo ma trận
    mat = df.pivot_table(index="Skill", columns="Member", values="Effort", fill_value=0.0)
    _apply_theme()
    plt.figure(figsize=(max(9, 1.2*len(mat.columns)), max(6, 0.6*len(mat.index)+2)))
    im = plt.imshow(mat.values, aspect="auto", cmap=mpl.colormaps.get("magma"))
    plt.colorbar(im, label="Effort")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=30, ha="right")
    plt.title("Effort Flow — Skill → Member (Top)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    return out_path
# ==== ADDITIONS ====
import math

def save_member_radar_grid(profiles: dict[str, pd.Series], out_path: str, max_cols: int = 3, limit: int = 9):
    """
    Vẽ lưới radar 3x3 (mặc định) cho top 'limit' members (theo tổng effort profile).
    """
    if not profiles: return None
    names = list(profiles.keys())[:limit]
    k = len(names)
    cols = min(max_cols, k)
    rows = math.ceil(k/cols)
    _apply_theme()
    fig = plt.figure(figsize=(cols*4.2, rows*4.2))
    # chuẩn trục
    labs = list(next(iter(profiles.values())).index)
    N = len(labs)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles_c = np.concatenate([angles, [angles[0]]])

    for idx, name in enumerate(names):
        vals = profiles[name].values.astype(float)
        vals_c = np.concatenate([vals, [vals[0]]])
        ax = fig.add_subplot(rows, cols, idx+1, polar=True)
        ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
        # nền vòng
        ax.set_ylim(0, 100); ax.set_yticks([20,40,60,80]); ax.set_yticklabels([])
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color=PALETTE["grid"])
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color=PALETTE["grid"])
        # polygon
        ax.plot(angles_c, vals_c, color=PALETTE["primary"], linewidth=1.8)
        ax.fill(angles_c, vals_c, color=PALETTE["primary"], alpha=0.22)
        # nhãn trục (rút gọn)
        ax.set_xticks(angles); ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(name, fontsize=11, pad=10, color=PALETTE["text"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    return out_path

def save_skill_timeline_area(expanded_df: pd.DataFrame, out_path: str):
    """
    Stack area: Effort theo Skill qua Week (toàn team).
    """
    if expanded_df is None or expanded_df.empty or "Skills" not in expanded_df.columns:
        return None
    # explode skills
    rows=[]
    for _, r in expanded_df.iterrows():
        eff = float(pd.to_numeric(r.get("Effort_Score", 1.0), errors="coerce") or 1.0)
        wk  = int(pd.to_numeric(r.get("Week", 1), errors="coerce") or 1)
        ss = r.get("Skills", "[]")
        if isinstance(ss, list): labels = ss
        else:
            try:
                import json; labels = json.loads(ss)
            except: labels = []
        if not labels: labels = ["General"]
        for lab in labels:
            rows.append((wk, str(lab), eff))
    df = pd.DataFrame(rows, columns=["Week","Skill","Effort"])
    piv = df.pivot_table(index="Week", columns="Skill", values="Effort", aggfunc="sum", fill_value=0.0).sort_index()
    if piv.empty: return None
    _apply_theme()
    plt.figure(figsize=(10,5.4))
    x = piv.index.values
    y = piv.values.T
    plt.stackplot(x, *y, labels=list(piv.columns))
    plt.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    plt.title("Skill Mix Over Time")
    plt.xlabel("Week"); plt.ylabel("Effort (sum)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    return out_path

def save_hhi_timeseries(expanded_df: pd.DataFrame, out_path: str):
    """
    HHI theo thời gian dựa trên phân phối Effort giữa các Skill.
    """
    if expanded_df is None or expanded_df.empty or "Skills" not in expanded_df.columns:
        return None
    rows=[]
    for _, r in expanded_df.iterrows():
        eff = float(pd.to_numeric(r.get("Effort_Score", 1.0), errors="coerce") or 1.0)
        wk  = int(pd.to_numeric(r.get("Week", 1), errors="coerce") or 1)
        ss = r.get("Skills", "[]")
        if isinstance(ss, list): labels = ss
        else:
            try:
                import json; labels = json.loads(ss)
            except: labels = []
        if not labels: labels = ["General"]
        for lab in labels:
            rows.append((wk, str(lab), eff))
    df = pd.DataFrame(rows, columns=["Week","Skill","Effort"])
    hhi=[]
    for wk, sub in df.groupby("Week"):
        s = sub.groupby("Skill")["Effort"].sum()
        p = s / s.sum()
        hhi.append((wk, float(np.sum(np.square(p.values)))))
    _apply_theme()
    plt.figure(figsize=(9,4.8))
    wks, vals = zip(*sorted(hhi))
    plt.plot(wks, vals, marker="o", linewidth=2.0, color=PALETTE["purple"])
    plt.title("Team Skill Concentration (HHI) Over Time")
    plt.xlabel("Week"); plt.ylabel("HHI (0..1)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    return out_path

def save_bus_factor_bar(df_members: pd.DataFrame, out_path: str, topn: int = 10):
    """
    Bar chart top key_person_index (rủi ro người-khoá).
    """
    if df_members is None or df_members.empty: return None
    d = df_members.head(topn)
    _apply_theme()
    plt.figure(figsize=(10, 4.8))
    bars = plt.bar(d["Member"], d["key_person_index"], color=PALETTE["danger"])
    plt.xticks(rotation=30, ha="right")
    for b, solo in zip(bars, d["solo_share"]):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.2f}\nsolo {solo:.0%}",
                 ha="center", va="bottom", color=PALETTE["text"], fontsize=8)
    plt.title("Key-Person Risk (Bus-Factor Proxy) — higher = riskier")
    plt.ylabel("Key Person Index")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    return out_path
