"""
report_generator.py — Exports evaluation results as CSV or PDF.
PDF generation uses ReportLab; falls back to plain text if unavailable.
"""

import io
import csv
import datetime
import os

import numpy as np


def export_csv(y_true: np.ndarray,
               y_pred: np.ndarray,
               metrics: dict,
               path: str) -> str:
    """
    Save actual vs predicted arrays plus metric summary to a CSV file.

    Returns the absolute path to the saved file.
    """
    path = _ensure_extension(path, ".csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Metric summary block
        writer.writerow(["=== Metrics ==="])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}"])
        writer.writerow([])
        # Data block
        writer.writerow(["Index", "Actual", "Predicted"])
        for i, (a, p) in enumerate(zip(y_true, y_pred)):
            writer.writerow([i, float(a), float(p)])
    return os.path.abspath(path)


def export_pdf(y_true: np.ndarray,
               y_pred: np.ndarray,
               metrics: dict,
               model_params: dict,
               path: str,
               fig=None) -> str:
    """
    Save a PDF report containing:
      - Title + timestamp
      - Model parameters table
      - Evaluation metrics table
      - Embedded comparison chart (if fig supplied)

    Falls back to a plain-text .txt file if ReportLab is not installed.

    Returns the absolute path to the saved file.
    """
    path = _ensure_extension(path, ".pdf")
    try:
        return _pdf_reportlab(y_true, y_pred, metrics, model_params, path, fig)
    except ImportError:
        # Graceful fallback: write a text report
        txt_path = path.replace(".pdf", "_report.txt")
        return _txt_fallback(metrics, model_params, txt_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_extension(path: str, ext: str) -> str:
    """Append extension if not already present."""
    if not path.lower().endswith(ext):
        path += ext
    return path


def _pdf_reportlab(y_true, y_pred, metrics, model_params, path, fig):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, Image
    )
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    story  = []
    W, H   = A4

    # Title
    title_style = styles["Title"]
    story.append(Paragraph("Solar Power Prediction — Evaluation Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.5 * cm))

    # Model parameters table
    story.append(Paragraph("Model Parameters", styles["Heading2"]))
    param_data = [["Parameter", "Value"]] + [
        [str(k), str(v)] for k, v in model_params.items()
    ]
    _add_table(story, param_data)
    story.append(Spacer(1, 0.4 * cm))

    # Metrics table
    story.append(Paragraph("Evaluation Metrics", styles["Heading2"]))
    metric_data = [["Metric", "Value"]] + [
        [k, f"{v:.6f}"] for k, v in metrics.items()
    ]
    _add_table(story, metric_data)
    story.append(Spacer(1, 0.4 * cm))

    # Embed chart
    if fig is not None:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            img = Image(buf, width=16 * cm, height=8 * cm)
            story.append(Paragraph("Actual vs Predicted", styles["Heading2"]))
            story.append(img)
        except Exception:
            pass

    doc = SimpleDocTemplate(path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)
    return os.path.abspath(path)


def _add_table(story, data):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)


def _txt_fallback(metrics, model_params, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Solar Power Prediction — Evaluation Report\n")
        f.write(f"Generated: {datetime.datetime.now()}\n\n")
        f.write("== Model Parameters ==\n")
        for k, v in model_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n== Metrics ==\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
    return os.path.abspath(path)
