"""Data ingestion module for loading and sanitizing CSV input files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class IngestionError(Exception):
    """Raised when ingestion operations fail."""

class SchemaMismatchError(IngestionError):
    """Raised when CSV schema does not match expected columns."""
    pass



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


def read_csv_safely(
    file_path: Path,
    expected_columns: list[str] | None = None,
) -> list[dict[str, Any]]:

    """Read CSV content with minimal sanitization and safety checks.

    Args:
        file_path: Path to the CSV file.
        required_columns: Optional set of required column names.

    Returns:
        List of sanitized row dictionaries.

    Raises:
        IngestionError: If CSV ingestion fails.
        SchemaMismatchError: If required columns are missing.
    """
    if not file_path.exists() or not file_path.is_file():
        raise IngestionError(f"CSV file not found: {file_path}")
    # Reject non-CSV extensions
    if file_path.suffix.lower() != ".csv":
        raise IngestionError("Only .csv files are allowed")

    rows: list[dict[str, Any]] = []

    try:
        with file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            # NEW: Schema validation
            if expected_columns is not None:
                actual_columns = set(reader.fieldnames or [])
                expected_set = set(expected_columns)
                if not expected_set.issubset(actual_columns):
                    missing = expected_set - actual_columns
                    raise SchemaMismatchError(
                        f"Missing required columns: {missing}"
                    )


            for row in reader:
                rows.append(sanitize_csv_row(row))

    except SchemaMismatchError:
        raise  # Preserve specific schema errors

    except (OSError, csv.Error) as exc:
        raise IngestionError("Failed to ingest CSV data") from exc

    return rows
