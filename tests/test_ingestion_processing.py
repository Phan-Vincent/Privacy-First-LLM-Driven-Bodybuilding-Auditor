"""Unit tests for ingestion and processing module error paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import IngestionError, SchemaMismatchError, read_csv_safely


def test_malformed_csv_rejected_via_schema_mismatch(tmp_path: Path) -> None:
    """CSV with missing header fields should be rejected."""

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("date,calories\n2026-01-01,2500\n", encoding="utf-8")

    with pytest.raises(SchemaMismatchError):
        read_csv_safely(bad_csv, expected_columns=["date", "calories", "protein_g", "bodyweight_lb"])


def test_non_csv_extension_rejected(tmp_path: Path) -> None:
    """Non-.csv file extension should raise IngestionError."""

    bad_file = tmp_path / "nutrition.txt"
    bad_file.write_text("date,calories\n", encoding="utf-8")

    with pytest.raises(IngestionError):
        read_csv_safely(bad_file, expected_columns=["date", "calories"])


def test_missing_required_columns_in_processing_raises_explicit_exception() -> None:
    """Processing should raise MissingColumnsError when required columns are absent."""

    pd = pytest.importorskip("pandas")
    from processing import MissingColumnsError, process_athlete_analytics

    nutrition_df = pd.DataFrame(
        [
            {"date": "2026-01-01", "calories": 2500, "bodyweight_lb": 200},
        ]
    )
    training_df = pd.DataFrame(
        [
            {"date": "2026-01-01", "muscle_group": "chest", "sets": 3, "reps": 8, "weight": 225},
        ]
    )

    with pytest.raises(MissingColumnsError):
        process_athlete_analytics(nutrition_df=nutrition_df, training_df=training_df)
