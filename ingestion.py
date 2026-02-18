"""Data ingestion module for loading and sanitizing CSV input files."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable


FORMULA_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


class IngestionError(Exception):
    """Raised when ingestion operations fail."""


class SchemaMismatchError(IngestionError):
    """Raised when CSV headers do not match the expected schema."""


def _sanitize_cell(value: str | None) -> str:
    """Sanitize a single CSV cell.

    Sanitization rules:
    - Remove null bytes.
    - Trim surrounding whitespace.
    - Strip spreadsheet formula prefixes.
    - Remove HTML tags.

    Args:
        value: Raw CSV cell value.

    Returns:
        Sanitized cell value.
    """
    normalized = (value or "").replace("\x00", "").strip()

    while normalized.startswith(FORMULA_PREFIXES):
        normalized = normalized[1:].lstrip()

    normalized = HTML_TAG_PATTERN.sub("", normalized)
    return normalized


def sanitize_csv_row(row: dict[str, str | None]) -> dict[str, str]:
    """Sanitize a CSV row by cleaning keys and values.

    Args:
        row: Raw CSV row mapping column names to values.

    Returns:
        A sanitized CSV row.
    """
    sanitized: dict[str, str] = {}
    for key, value in row.items():
        safe_key = _sanitize_cell(key)
        safe_value = _sanitize_cell(value)
        sanitized[safe_key] = safe_value
    return sanitized


def _validate_extension(file_path: Path) -> None:
    """Validate that the file path points to a CSV file.

    Args:
        file_path: Path to validate.

    Raises:
        IngestionError: If the path is not a CSV file.
    """
    if file_path.suffix.lower() != ".csv":
        raise IngestionError(f"Invalid file extension for '{file_path.name}'. Expected .csv")


def _validate_schema(actual_columns: list[str], expected_columns: Iterable[str]) -> None:
    """Validate CSV schema against expected columns.

    Args:
        actual_columns: Header columns discovered in the CSV.
        expected_columns: Required columns expected by the pipeline.

    Raises:
        SchemaMismatchError: If expected and actual headers differ.
    """
    expected = [_sanitize_cell(column) for column in expected_columns]
    actual = [_sanitize_cell(column) for column in actual_columns]

    if expected != actual:
        missing = [col for col in expected if col not in actual]
        unexpected = [col for col in actual if col not in expected]
        raise SchemaMismatchError(
            "CSV schema mismatch. "
            f"Expected columns: {expected}. "
            f"Actual columns: {actual}. "
            f"Missing: {missing or 'none'}. "
            f"Unexpected: {unexpected or 'none'}."
        )


def read_csv_safely(file_path: Path, expected_columns: list[str]) -> list[dict[str, Any]]:
    """Read CSV content with sanitization and strict schema checks.

    Args:
        file_path: Path to the CSV file.
        expected_columns: Ordered list of required column names.

    Returns:
        List of sanitized row dictionaries.

    Raises:
        IngestionError: If CSV ingestion fails.
        SchemaMismatchError: If the CSV schema does not match expected columns.
    """
    if not file_path.exists() or not file_path.is_file():
        raise IngestionError(f"CSV file not found: {file_path}")

    _validate_extension(file_path)

    rows: list[dict[str, Any]] = []
    try:
        with file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SchemaMismatchError("CSV file is missing a header row")

            _validate_schema(reader.fieldnames, expected_columns)

            for row in reader:
                rows.append(sanitize_csv_row(row))
    except SchemaMismatchError:
        raise
    except (OSError, csv.Error) as exc:
        raise IngestionError("Failed to ingest CSV data") from exc

    return rows
