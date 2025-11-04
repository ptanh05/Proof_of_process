import argparse
import os
import sys
import pandas as pd

from ..pipeline.orchestrator import run_pipeline
from ..data.synthetic_data_factory import make as make_synth
from ..connectors.trello_connector import from_csv as trello_csv
from ..utils.logging_utils import get_logger

log = get_logger()


def _maybe_convert_gsheet_view_to_export(url: str) -> str:
    """
    Chuyển các URL Google Sheets dạng VIEW/EDIT thành URL EXPORT CSV:
    Hỗ trợ các dạng:
      - https://docs.google.com/spreadsheets/d/<ID>/edit?usp=sharing
      - https://docs.google.com/spreadsheets/d/<ID>/edit?gid=<GID>#gid=<GID>
    Quy tắc:
      - Lấy <ID> từ '/spreadsheets/d/<ID>'
      - Ưu tiên gid ở cuối (anchor '#gid=...') nếu có; nếu không, lấy từ query 'gid=...'
      - Nếu không có gid -> trả về export CSV không kèm gid (Google sẽ trả tab đầu — thường là gid=0)
    """
    try:
        if "docs.google.com/spreadsheets/d/" not in url:
            return url

        import re
        # Bắt ID
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9\-_]+)", url)
        if not m:
            return url
        sheet_id = m.group(1)

        # Ưu tiên gid trong anchor (sau #), nếu có nhiều thì lấy cái cuối
        anchor_gids = re.findall(r"#gid=([0-9]+)", url)
        gid = anchor_gids[-1] if anchor_gids else None

        # Nếu chưa có, thử lấy từ query (?gid=....)
        if gid is None:
            query_gids = re.findall(r"[?&]gid=([0-9]+)", url)
            gid = query_gids[-1] if query_gids else None

        # Lắp URL export CSV
        base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        if gid:
            return f"{base}&gid={gid}"
        # Không có gid -> export tab đầu
        return base
    except Exception:
        # có lỗi gì thì trả về nguyên URL cũ (để pandas tự xử, sẽ lỗi 404 nếu sheet private/định dạng không đúng)
        return url


def _read_csv_safely(path: str, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """
    Đọc CSV từ file path hoặc URL (kể cả Google Sheets).
    - Tự động chuyển VIEW/EDIT URL -> EXPORT CSV khi là Google Sheets.
    - Hỗ trợ STDIN khi path == "-"
    """
    try:
        if path == "-":
            log.info("Reading CSV from STDIN...")
            return pd.read_csv(sys.stdin, sep=sep, encoding=encoding)

        # Chuẩn hoá URL Google Sheets nếu có
        if isinstance(path, str) and "docs.google.com/spreadsheets/d/" in path:
            norm = _maybe_convert_gsheet_view_to_export(path)
            if norm != path:
                log.info(f"Google Sheets URL normalized to CSV export: {norm}")
                path = norm

        log.info(f"Reading CSV from: {path}")
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as e:
        log.error(f"Failed to read CSV [{path}]: {e}")
        raise




def _load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Ưu tiên nguồn dữ liệu theo thứ tự:
      1) --demo
      2) --trello-csv
      3) --csv (file/URL/'-')
      4) --gsheet-id (+ --cred) [auto-tab hoặc tab chỉ định]
    """
    # 1) DEMO
    if args.demo:
        df = make_synth(n_tasks=args.demo_tasks)
        log.info(f"Using synthetic demo dataset (n_tasks={len(df)})")
        return df

    # 2) TRELLO CSV
    if args.trello_csv:
        df = trello_csv(args.trello_csv, project_name=args.project_name or "TrelloProject")
        log.info(f"Loaded Trello CSV. shape={df.shape}")
        return df

    # 3) CSV (local file / URL / stdin)
    if args.csv:
        df = _read_csv_safely(args.csv, sep=args.sep, encoding=args.encoding)
        log.info(f"Loaded CSV. shape={df.shape}")
        return df

    # 4) Google Sheets API
    if args.gsheet_id and args.cred:
        try:
            from ..connectors.google_sheets_connector import read_sheet, read_sheet_auto
        except Exception as e:
            log.error(
                "Google Sheets connector not available. "
                "Install extras: pip install gspread google-auth"
            )
            raise

        if args.gsheet_auto:
            df, picked = read_sheet_auto(args.gsheet_id, args.cred)
            log.info(f"Loaded Google Sheet via API (auto tab='{picked}'). shape={df.shape}")
        else:
            tab = args.gsheet_tab or "Sheet1"
            df = read_sheet(args.gsheet_id, tab, args.cred)
            log.info(f"Loaded Google Sheet via API (tab='{tab}'). shape={df.shape}")
        return df

    # Không có nguồn nào được cung cấp
    raise SystemExit(
        "Please provide one data source: "
        "--demo OR --trello-csv PATH OR --csv PATH_OR_URL_OR_- OR --gsheet-id + --cred"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Proof of Process — Run end-to-end analytics pipeline"
    )

    # Nguồn dữ liệu
    ap.add_argument("--demo", action="store_true", help="Use synthetic dataset")
    ap.add_argument("--demo-tasks", type=int, default=250, help="Number of synthetic tasks")

    ap.add_argument("--trello-csv", type=str, help="Path to Trello CSV export")
    ap.add_argument("--project-name", type=str, help="Project name when using Trello CSV")

    ap.add_argument("--csv", type=str, help="Path/URL to CSV with tasks (use '-' for STDIN)")
    ap.add_argument("--sep", type=str, default=",", help="CSV separator (default ',')")
    ap.add_argument("--encoding", type=str, default="utf-8", help="CSV encoding (default utf-8)")

    ap.add_argument("--gsheet-id", type=str, help="Google Sheet ID (API mode)")
    ap.add_argument("--gsheet-tab", type=str, help="Worksheet name (optional if --gsheet-auto)")
    ap.add_argument("--gsheet-auto", action="store_true", help="Auto-pick the best worksheet/tab")
    ap.add_argument("--cred", type=str, help="Path to service account JSON for Google Sheets API")

    # Output
    ap.add_argument("--out", type=str, default="reports", help="Output folder")

    args = ap.parse_args()

    # Load data
    df = _load_dataframe(args)

    # Kiểm tra tối thiểu
    if df is None or df.empty:
        raise SystemExit("Loaded DataFrame is empty. Please check your data source.")
    log.info(f"Data loaded OK. rows={len(df):,} cols={len(df.columns)}")

    # Run pipeline
    res = run_pipeline(df, outdir=args.out)
    log.info("=== EXECUTIVE SUMMARY ===")
    for k, v in res.get("summary", {}).items():
        log.info(f"{k}: {v}")
    log.info("Pipeline finished.")


if __name__ == "__main__":
    main()
