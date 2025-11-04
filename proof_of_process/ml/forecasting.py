import numpy as np, pandas as pd

def monte_carlo_eta(remaining_effort: float, past_velocity: np.ndarray, trials=5000, max_weeks=104):
    """
    Simulate weekly completion until remaining <= 0
    Returns distribution of weeks_to_finish
    """
    if remaining_effort <= 0:
        return np.zeros(trials)
    if len(past_velocity)==0 or past_velocity.sum()==0:
        past_velocity = np.array([1.0])
    weeks = np.zeros(trials, dtype=int)
    for t in range(trials):
        rem = remaining_effort
        w=0
        while rem>0 and w<max_weeks:
            draw = np.random.choice(past_velocity)
            draw = max(draw, 0.1)
            rem -= draw
            w+=1
        weeks[t]=w
    return weeks

def summarize_eta(weeks_arr: np.ndarray):
    p10 = np.percentile(weeks_arr, 10)
    p50 = np.percentile(weeks_arr, 50)
    p90 = np.percentile(weeks_arr, 90)
    return {"weeks_p10": float(p10), "weeks_p50": float(p50), "weeks_p90": float(p90)}
