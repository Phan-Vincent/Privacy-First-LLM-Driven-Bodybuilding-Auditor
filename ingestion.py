"""Data ingestion module for loading and sanitizing CSV input files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class IngestionError(Exception):
    """Raised when ingestion operations fail."""


def sanitize_csv_row(row: dict[str, str]) -> dict[str, str]:
    """Sanitize a CSV row by trimming whitespace and removing null characters.

    Args:
        row: Raw CSV row mapping column names to values.

    Returns:
        A sanitized CSV row.
    """
    sanitized: dict[str, str] = {}
    for key, value in row.items():
        safe_key = key.strip().replace("\x00", "") if key is not None else ""
        safe_value = value.strip().replace("\x00", "") if value is not None else ""
        sanitized[safe_key] = safe_value
    return sanitized


def read_csv_safely(file_path: Path) -> list[dict[str, Any]]:
    """Read CSV content with minimal sanitization and safety checks.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of sanitized row dictionaries.

    Raises:
        IngestionError: If CSV ingestion fails.
    """
    if not file_path.exists() or not file_path.is_file():
        raise IngestionError(f"CSV file not found: {file_path}")

    rows: list[dict[str, Any]] = []
    try:
        with file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(sanitize_csv_row(row))
    except (OSError, csv.Error) as exc:
        raise IngestionError("Failed to ingest CSV data") from exc

    return rows
