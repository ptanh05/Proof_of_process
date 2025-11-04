# Proof of Process — Enterprise-grade Project Analytics (PoP)

Pipeline end-to-end:
- Ingest (CSV/Trello/Jira stubs) → Schema map → Member dedupe → Status normalize
- NLP Skill tagging (VN/EN), Skill leaderboard, Skill diversity
- Effort Estimator (Quantile GBM q10/q50/q90 + Conformal)
- EVM (PV/EV/AC) + SPI/CPI + Velocity/CFD/Burndown
- Monte Carlo ETA forecast (fan chart) + Risk score (logit over SPI/overdue/scope creep/Gini)
- Shapley Contribution % (sampling) + Gini/Theil
- Charts (matplotlib) + Executive PDF (ReportLab)

## Quickstart

1) Create virtual env & install (editable):
```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

Optional extras (Google Sheets API mode):
```bash
pip install -e .[gsheets]
```

2) Run with demo data:
```bash
pop-run-pipeline --demo --demo-tasks 250 --out reports
```

3) Ingest Google Sheet/XLSX → staging → run pipeline:
```bash
pop-ingest-run --in "<URL_or_.xlsx>" --staging staging/timeline --out reports
```

Notes:
- If using Google Sheets API with `--gsheet-id` in `pop-run-pipeline`, provide `--cred` pointing to a service account JSON and install extras `.[gsheets]`.
- Outputs include: `reports/Executive_Report.pdf`, charts in `reports/fig/`, and CSV exports in `reports/exports/`.
