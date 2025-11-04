# proof_of_process/cli/ingest_then_run.py
# v2.0 — Ingest trực tuyến → staging → chạy pipeline và xuất báo cáo
# - Nhận --out để chỉ thư mục xuất PDF/PNG/CSV
# - Truyền preview_rows xuống ingestor
# - Đọc external_input.csv trong staging rồi gọi orchestrator.run_pipeline(df, outdir)

import argparse
import json
import os
import sys
import pandas as pd

from ..ingest.ai_xlsx_ingestor import ingest_xlsx_to_dir
from ..pipeline.orchestrator import run_pipeline

try:
    from ..utils.logging_utils import get_logger
    log = get_logger()
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    log = logging.getLogger("ingest_then_run")


def main():
    ap = argparse.ArgumentParser(
        description="Ingest Google Sheet/XLSX → staging → Run pipeline → Xuất báo cáo (PDF/PNG/CSV)."
    )
    ap.add_argument("--in", dest="in_path", required=True,
                    help="URL Google Sheet hoặc đường dẫn .xlsx")
    ap.add_argument("--staging", dest="staging_dir", required=True,
                    help="Thư mục staging để xuất dữ liệu trung gian")
    ap.add_argument("--sheet", dest="sheet", default=None,
                    help="Tên sheet muốn ép dùng (tùy chọn)")
    ap.add_argument("--preview-rows", dest="preview_rows", type=int, default=200,
                    help="Số dòng preview HTML (mặc định 200)")
    ap.add_argument("--out", dest="outdir", required=True,
                    help="Thư mục xuất báo cáo (PDF/PNG/CSV)")

    args = ap.parse_args()

    # 1) INGEST → STAGING
    meta = ingest_xlsx_to_dir(
        args.in_path,
        args.staging_dir,
        sheet=args.sheet,
        preview_rows=args.preview_rows
    )
    # In ra JSON cho dễ debug trong CI/terminal
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    # 2) ĐỌC STAGING → DF
    ext_csv = os.path.join(args.staging_dir, "external_input.csv")
    if not os.path.exists(ext_csv):
        log.error("Không tìm thấy staged CSV: %s", ext_csv)
        sys.exit(2)

    try:
        df = pd.read_csv(ext_csv)
    except Exception as e:
        log.error("Không đọc được %s: %s", ext_csv, e)
        sys.exit(3)

    # 3) RUN PIPELINE → BÁO CÁO
    os.makedirs(args.outdir, exist_ok=True)
    try:
        res = run_pipeline(df, outdir=args.outdir)
    except TypeError:
        # fallback nếu bản orchestrator cũ có chữ ký khác
        res = run_pipeline(df, args.outdir)

    # 4) LOG & KẾT LUẬN
    report_path = None
    if isinstance(res, dict):
        report_path = (res.get("report_path")
                       or res.get("pdf_path")
                       or os.path.join(args.outdir, "Executive_Report.pdf"))

    if report_path and os.path.exists(report_path):
        log.info("Report saved: %s", report_path)
    else:
        log.info("Pipeline finished. Kiểm tra thư mục: %s", args.outdir)

    # In gọn summary nếu có
    try:
        print(res)
    except Exception:
        pass


if __name__ == "__main__":
    main()
