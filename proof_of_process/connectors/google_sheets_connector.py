import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from typing import Tuple, Optional, List
from ..ingest.schema_autodetect import ALIASES

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

REQUIRED = ["Task","Project","Description","Assignees","Status","Week"]

def _score_header_row(candidates: List[str]) -> int:
    """
    Chấm điểm 1 dòng header theo mức phủ REQUIRED và alias.
    +2 điểm: match tên chuẩn; +1 điểm: match alias chứa trong text.
    """
    score = 0
    low = [c.strip().lower() for c in candidates if c is not None]
    for need in REQUIRED:
        # match exact
        if any(c == need.lower() for c in low):
            score += 2
            continue
        # match alias chứa trong text
        aliases = [a.lower() for a in ALIASES.get(need, [])]
        if any(any(a in c for a in aliases) for c in low):
            score += 1
    return score

def _detect_header_and_dataframe(raw_values: List[List[str]]) -> Tuple[pd.DataFrame, int, int]:
    """
    Từ sheet values (mảng 2D), quét <= 30 dòng đầu để tìm header tốt nhất.
    Trả về (DataFrame, header_row_index, score).
    """
    if not raw_values:
        return pd.DataFrame(), -1, 0
    max_rows = min(30, len(raw_values))
    best = (-1, -1)  # (row_index, score)
    for r in range(max_rows):
        row = raw_values[r]
        sc = _score_header_row(row)
        if sc > best[1]:
            best = (r, sc)
    header_idx, score = best
    if header_idx < 0:
        # không tìm được header hợp lệ, fallback coi dòng 0 là header
        header_idx, score = 0, 0
    # tạo DataFrame từ header_idx
    headers = [h if h else f"col_{i}" for i, h in enumerate(raw_values[header_idx])]
    data = raw_values[header_idx+1:]
    df = pd.DataFrame(data, columns=headers)
    return df, header_idx, score

def _sheet_score(df_preview: pd.DataFrame) -> int:
    """
    Chấm điểm cả tab dựa trên số cột REQUIRED hiện diện sau map alias thô ở header.
    """
    if df_preview.empty:
        return 0
    cols = [c.lower() for c in df_preview.columns]
    score = 0
    for need in REQUIRED:
        if need.lower() in cols:
            score += 2
            continue
        aliases = [a.lower() for a in ALIASES.get(need, [])]
        if any(any(a in c for a in aliases) for c in cols):
            score += 1
    # khuyến khích tab có nhiều dữ liệu
    score += min(len(df_preview), 200) // 10  # +1 mỗi 10 dòng, tối đa +20
    return score

def read_sheet(spreadsheet_id: str, worksheet_name: str, cred_json_path: str) -> pd.DataFrame:
    """
    Đọc tab chỉ định (worksheet_name). Tự động phát hiện header row.
    """
    creds = Credentials.from_service_account_file(cred_json_path, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)
    raw = ws.get_all_values()
    df, header_row, _ = _detect_header_and_dataframe(raw)
    return df

def read_sheet_auto(spreadsheet_id: str, cred_json_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Tự động chọn tab tốt nhất & phát hiện header row.
    Trả về (DataFrame, worksheet_name_chosen).
    """
    creds = Credentials.from_service_account_file(cred_json_path, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    candidates = []
    for ws in sh.worksheets():
        raw = ws.get_all_values()
        df_preview, hdr_idx, hdr_score = _detect_header_and_dataframe(raw)
        tab_score = _sheet_score(df_preview)
        total = hdr_score + tab_score
        candidates.append((total, ws.title, df_preview))
    if not candidates:
        return pd.DataFrame(), ""
    # chọn ứng viên điểm cao nhất
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_df = candidates[0]
    return best_df, best_name
