from __future__ import annotations
import numpy as np

def _eta_from_velocity(remaining_effort: float, velocity_mean: float) -> float:
    v = max(velocity_mean, 1e-6)
    return float(np.ceil(remaining_effort / v))

def run_scenarios(remaining_effort: float, velocity: np.ndarray) -> dict:
    v = np.asarray(velocity, dtype=float)
    mu = float(np.nanmean(v[v>0])) if np.any(v>0) else 0.0
    base_eta = _eta_from_velocity(remaining_effort, mu) if mu>0 else None
    out = {"baseline_eta_weeks": base_eta, "scenarios": []}
    if mu>0:
        for name, factor in [("velocity+10%", 1.10), ("velocity+25%", 1.25), ("scope-20%", None)]:
            if factor is not None:
                eta = _eta_from_velocity(remaining_effort, mu*factor)
            else:
                eta = _eta_from_velocity(remaining_effort*0.8, mu)
            out["scenarios"].append({"name": name, "eta_weeks": float(eta), "delta_vs_base": float(eta-base_eta)})
    return out
