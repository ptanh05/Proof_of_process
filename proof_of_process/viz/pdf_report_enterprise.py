# proof_of_process/viz/pdf_report_enterprise.py
from __future__ import annotations
import os, datetime
from typing import Dict, List
import pandas as pd

PRIMARY  = "#5B8FF9"
ACCENT   = "#5AD8A6"
MUTED    = "#A7B0C8"
DARK     = "#0B1021"
PANEL    = "#12172B"
INK      = "#0E122B"

def build_pdf(title: str,
              summary_df: pd.DataFrame,
              charts: List[str],
              tables: Dict[str, pd.DataFrame],
              out_path: str):
    """
    PDF 'executive':
      - Cover (title + timestamp)
      - Header/Footer với số trang
      - KPI cards (grid 3 cột)
      - Charts (bố cục 2 cột)
      - Tables (KPIs chi tiết...)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Image, Table, TableStyle, PageBreak)
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        def _header_footer(c: canvas.Canvas, doc):
            c.setFillColor(colors.HexColor(INK))
            c.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
            # Header bar
            c.setFillColor(colors.HexColor(PANEL))
            c.rect(0, doc.pagesize[1]-1.4*cm, doc.pagesize[0], 1.4*cm, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#EAF0FF"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1.5*cm, doc.pagesize[1]-0.9*cm, "Proof of Process — Executive Report")
            # Footer
            c.setFillColor(colors.HexColor(PANEL))
            c.rect(0, 0, doc.pagesize[0], 1.0*cm, fill=1, stroke=0)
            c.setFillColor(colors.HexColor(MUTED))
            c.setFont("Helvetica", 9)
            c.drawRightString(doc.pagesize[0]-1.2*cm, 0.5*cm, f"Page {doc.page}")

        doc = SimpleDocTemplate(out_path, pagesize=A4,
                                leftMargin=1.5*cm, rightMargin=1.5*cm,
                                topMargin=2.6*cm, bottomMargin=1.6*cm)

        styles = getSampleStyleSheet()
        H1 = ParagraphStyle("H1", parent=styles["Title"], fontName="Helvetica-Bold",
                            fontSize=22, textColor=colors.HexColor("#EAF0FF"))
        H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                            textColor=colors.white, backColor=colors.HexColor(PANEL), spaceAfter=6)
        P  = ParagraphStyle("P", parent=styles["BodyText"], fontName="Helvetica",
                            textColor=colors.HexColor("#EAF0FF"))

        story: List = []
        # Cover
        story += [Paragraph(title, H1), Spacer(1, 8)]
        story += [Paragraph(datetime.datetime.now().strftime("Generated on %Y-%m-%d %H:%M"), P),
                  Spacer(1, 24)]
        story += [Paragraph("Executive Summary", H2), Spacer(1, 4)]

        # KPI cards (3 cột x 2 hàng)
        sd = summary_df.T.reset_index()
        sd.columns = ["Metric", "Value"]

        def _fmt(v):
            try:
                f = float(v)
                if abs(f) >= 1000: return f"{f/1000:.1f}k"
                return f"{f:.2f}"
            except Exception:
                return str(v)
        sd["Value"] = sd["Value"].map(_fmt)

        data, row = [], []
        for i, r in sd.iterrows():
            cell = [[Paragraph(f"<b>{r['Metric']}</b>", P)],
                    [Paragraph(f"<para color='{ACCENT}'><b>{r['Value']}</b></para>", P)]]
            tbl = Table(cell, colWidths=[6.2*cm], rowHeights=[1.0*cm, 0.9*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1), colors.HexColor(PANEL)),
                ("BOX",(0,0),(-1,-1), 0.7, colors.HexColor(PRIMARY)),
                ("LEFTPADDING",(0,0),(-1,-1), 6),
                ("RIGHTPADDING",(0,0),(-1,-1), 6),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            row.append(tbl)
            if (i+1) % 3 == 0:
                data.append(row); row = []
        if row: data.append(row)
        for r in data:
            t = Table([r], colWidths=[6.2*cm]*len(r))
            t.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"LEFT")]))
            story += [t, Spacer(1, 8)]

        # Charts (2 cột)
        if charts:
            story += [Spacer(1, 8), Paragraph("Charts", H2), Spacer(1, 8)]
            pair = []
            for p in charts:
                if p and os.path.exists(p):
                    pair.append(Image(p, width=250, height=160))
                    if len(pair) == 2:
                        story += [Table([pair], colWidths=[8*cm, 8*cm]), Spacer(1, 8)]
                        pair = []
            if pair:
                story += [pair[0], Spacer(1, 8)]

        # Tables
        for name, df in (tables or {}).items():
            if df is None or df.empty:
                continue
            story += [PageBreak(), Paragraph(name, H2), Spacer(1, 6)]
            dat = [df.columns.tolist()] + df.astype(str).values.tolist()
            tt = Table(dat, hAlign="LEFT", colWidths=[4*cm] + [11*cm])
            tt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0), colors.HexColor(PRIMARY)),
                ("TEXTCOLOR",(0,0),(-1,0), colors.white),
                ("GRID",(0,0),(-1,-1), 0.25, colors.HexColor(MUTED)),
                ("BACKGROUND",(0,1),(-1,-1), colors.HexColor(PANEL)),
                ("TEXTCOLOR",(0,1),(-1,-1), colors.HexColor("#EAF0FF")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor(PANEL), colors.HexColor("#0F1530")]),
                ("LEFTPADDING",(0,0),(-1,-1), 6),
                ("RIGHTPADDING",(0,0),(-1,-1), 6),
            ]))
            story += [tt]

        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
        return out_path

    except Exception as e:
        # Fallback: nếu reportlab lỗi, vẫn xuất tóm tắt
        base = os.path.splitext(out_path)[0]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        summary_df.to_csv(base + "_summary.csv", index=False)
        with open(base + "_fallback.txt", "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{e}\n")
        return base + "_fallback.txt"
