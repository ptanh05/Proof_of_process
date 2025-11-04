from dataclasses import dataclass

@dataclass
class Config:
    random_state: int = 42
    n_jobs: int = -1
    min_rows: int = 50
    shapley_samples: int = 200  # permutation samples/task for Shapley approx
    forecast_history_weeks: int = 12
    monte_carlo_trials: int = 5000
    report_title: str = "Proof of Process — Executive Report"
