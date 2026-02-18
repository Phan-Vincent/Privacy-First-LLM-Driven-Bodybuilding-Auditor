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
