"""Reporting module for serializing audit results into output artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportError(Exception):
    """Raised when report generation fails."""


def write_json_report(report_data: dict[str, Any], output_path: Path) -> None:
    """Write a structured JSON report to disk.

    Args:
        report_data: Structured audit report data.
        output_path: Destination path for the report file.

    Raises:
        ReportError: If report serialization or write fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report_data, handle, indent=2, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ReportError("Failed to write JSON report") from exc


def _escape_pdf_text(text: str) -> str:
    """Escape PDF text characters for literal strings."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf_report(report_data: dict[str, Any], output_path: Path) -> None:
    """Write a minimal single-page PDF report.

    Args:
        report_data: Structured report payload.
        output_path: Destination path for the PDF file.

    Raises:
        ReportError: If PDF generation fails.
    """
    try:
        summary_text = str(report_data.get("audit", {}).get("summary", "No summary available"))
        lines = [
            "Athlete Audit Report",
            "",
            f"Summary: {summary_text}",
        ]

        y_position = 760
        line_commands: list[str] = []
        for line in lines:
            escaped = _escape_pdf_text(line)
            line_commands.append(f"BT /F1 12 Tf 50 {y_position} Td ({escaped}) Tj ET")
            y_position -= 20

        content_stream = "\n".join(line_commands).encode("latin-1", errors="replace")
        pdf = bytearray()
        xref_positions: list[int] = []

        def _append(obj: bytes) -> None:
            xref_positions.append(len(pdf))
            pdf.extend(obj)

        pdf.extend(b"%PDF-1.4\n")
        _append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
        _append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
        _append(
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        )
        _append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
        _append(
            f"5 0 obj << /Length {len(content_stream)} >> stream\n".encode("latin-1")
            + content_stream
            + b"\nendstream\nendobj\n"
        )

        xref_start = len(pdf)
        pdf.extend(f"xref\n0 {len(xref_positions) + 1}\n".encode("latin-1"))
        pdf.extend(b"0000000000 65535 f \n")
        for position in xref_positions:
            pdf.extend(f"{position:010d} 00000 n \n".encode("latin-1"))
        pdf.extend(
            f"trailer << /Size {len(xref_positions) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
                "latin-1"
            )
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(pdf))
    except OSError as exc:
        raise ReportError("Failed to write PDF report") from exc
