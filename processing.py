"""Deterministic analytics processing for nutrition and training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class ProcessingError(Exception):
    """Raised when processing operations fail."""


class MissingColumnsError(ProcessingError):
    """Raised when required columns are missing from an input dataframe."""


@dataclass(frozen=True)
class RequiredColumns:
    """Container for required dataframe column sets."""

    nutrition: tuple[str, ...] = ("date", "calories", "protein_g", "bodyweight_lb")
    training: tuple[str, ...] = ("date", "muscle_group", "sets", "reps")


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], frame_name: str) -> None:
    """Raise a clear error if required columns are missing.

    Args:
        frame: DataFrame to validate.
        required: Required column names.
        frame_name: Human-readable frame name for error messages.

    Raises:
        MissingColumnsError: If one or more required columns are absent.
    """
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MissingColumnsError(f"{frame_name} missing required columns: {missing}")


def _to_week_start(frame: pd.DataFrame, date_column: str = "date") -> pd.Series:
    """Convert date column to week start timestamps.

    Args:
        frame: Source dataframe.
        date_column: Name of the date column.

    Returns:
        Series of week-start timestamps.

    Raises:
        ProcessingError: If date conversion fails.
    """
    try:
        parsed_dates = pd.to_datetime(frame[date_column], errors="raise", utc=False)
    except (KeyError, ValueError, TypeError) as exc:
        raise ProcessingError(f"Invalid or missing date column '{date_column}'") from exc
    return parsed_dates.dt.to_period("W").dt.start_time


def compute_weekly_nutrition_metrics(nutrition_df: pd.DataFrame, calorie_target: float | None = None) -> dict[str, Any]:
    """Compute weekly nutrition metrics.

    Args:
        nutrition_df: Cleaned nutrition dataframe.
        calorie_target: Optional daily calorie target.

    Returns:
        Dictionary containing weekly nutrition metrics.
    """
    required = RequiredColumns().nutrition
    _require_columns(nutrition_df, required, "nutrition dataframe")

    frame = nutrition_df.copy()
    frame["week_start"] = _to_week_start(frame)

    weekly = (
        frame.groupby("week_start", as_index=False)
        .agg(
            avg_calories=("calories", "mean"),
            avg_protein_g=("protein_g", "mean"),
            avg_bodyweight_lb=("bodyweight_lb", "mean"),
        )
        .sort_values("week_start")
    )
    weekly["protein_per_lb"] = weekly["avg_protein_g"] / weekly["avg_bodyweight_lb"]

    if calorie_target is not None:
        if calorie_target <= 0:
            raise ProcessingError("calorie_target must be greater than zero")
        weekly["calorie_adherence_pct"] = (weekly["avg_calories"] / calorie_target) * 100.0
    else:
        weekly["calorie_adherence_pct"] = pd.NA

    latest = weekly.iloc[-1] if not weekly.empty else None

    return {
        "weekly": [
            {
                "week_start": row["week_start"].date().isoformat(),
                "avg_calories": float(row["avg_calories"]),
                "avg_protein_g": float(row["avg_protein_g"]),
                "avg_bodyweight_lb": float(row["avg_bodyweight_lb"]),
                "protein_per_lb": float(row["protein_per_lb"]),
                "calorie_adherence_pct": (
                    None if pd.isna(row["calorie_adherence_pct"]) else float(row["calorie_adherence_pct"])
                ),
            }
            for _, row in weekly.iterrows()
        ],
        "latest": (
            None
            if latest is None
            else {
                "week_start": latest["week_start"].date().isoformat(),
                "avg_calories": float(latest["avg_calories"]),
                "avg_protein_g": float(latest["avg_protein_g"]),
                "avg_bodyweight_lb": float(latest["avg_bodyweight_lb"]),
                "protein_per_lb": float(latest["protein_per_lb"]),
                "calorie_adherence_pct": (
                    None if pd.isna(latest["calorie_adherence_pct"]) else float(latest["calorie_adherence_pct"])
                ),
            }
        ),
    }


def compute_weekly_training_metrics(training_df: pd.DataFrame) -> dict[str, Any]:
    """Compute weekly training metrics and progression.

    Args:
        training_df: Cleaned training dataframe.

    Returns:
        Dictionary containing weekly training metrics.
    """
    required = RequiredColumns().training
    _require_columns(training_df, required, "training dataframe")

    frame = training_df.copy()
    frame["week_start"] = _to_week_start(frame)

    if "weight" not in frame.columns:
        frame["weight"] = 0.0

    frame["set_volume"] = frame["sets"].astype(float) * frame["reps"].astype(float) * frame["weight"].astype(float)
    frame["top_set_load"] = frame["reps"].astype(float) * frame["weight"].astype(float)

    weekly_volume = (
        frame.groupby(["week_start", "muscle_group"], as_index=False)
        .agg(total_volume=("set_volume", "sum"))
        .sort_values(["week_start", "muscle_group"])
    )

    weekly_top_set = (
        frame.groupby("week_start", as_index=False)
        .agg(current_week_top_set=("top_set_load", "max"))
        .sort_values("week_start")
    )
    weekly_top_set["prior_week_top_set"] = weekly_top_set["current_week_top_set"].shift(1)
    weekly_top_set["strength_velocity_pct"] = (
        (weekly_top_set["current_week_top_set"] - weekly_top_set["prior_week_top_set"])
        / weekly_top_set["prior_week_top_set"]
    ) * 100.0

    return {
        "weekly_volume_by_muscle_group": [
            {
                "week_start": row["week_start"].date().isoformat(),
                "muscle_group": str(row["muscle_group"]),
                "total_volume": float(row["total_volume"]),
            }
            for _, row in weekly_volume.iterrows()
        ],
        "strength_progression": [
            {
                "week_start": row["week_start"].date().isoformat(),
                "current_week_top_set": float(row["current_week_top_set"]),
                "prior_week_top_set": (None if pd.isna(row["prior_week_top_set"]) else float(row["prior_week_top_set"])),
                "strength_velocity_pct": (
                    None if pd.isna(row["strength_velocity_pct"]) else float(row["strength_velocity_pct"])
                ),
            }
            for _, row in weekly_top_set.iterrows()
        ],
    }


def detect_risk_flags(nutrition_metrics: dict[str, Any], training_metrics: dict[str, Any]) -> list[str]:
    """Detect risk flags from derived nutrition and training metrics.

    Args:
        nutrition_metrics: Output of ``compute_weekly_nutrition_metrics``.
        training_metrics: Output of ``compute_weekly_training_metrics``.

    Returns:
        List of risk flag strings.
    """
    flags: list[str] = []

    latest_nutrition = nutrition_metrics.get("latest")
    if latest_nutrition is not None:
        protein_per_lb = latest_nutrition.get("protein_per_lb")
        if isinstance(protein_per_lb, (int, float)) and protein_per_lb < 0.8:
            flags.append("Protein intake below 0.8 g/lb")

        adherence = latest_nutrition.get("calorie_adherence_pct")
        if isinstance(adherence, (int, float)) and abs(100.0 - adherence) > 15.0:
            flags.append("Calorie variance greater than 15%")

    volume_entries = training_metrics.get("weekly_volume_by_muscle_group", [])
    volume_df = pd.DataFrame(volume_entries)
    if not volume_df.empty:
        volume_df["week_start"] = pd.to_datetime(volume_df["week_start"], errors="raise")
        volume_df = volume_df.sort_values(["muscle_group", "week_start"])
        volume_df["prior_volume"] = volume_df.groupby("muscle_group")["total_volume"].shift(1)
        volume_df["volume_change_pct"] = (
            (volume_df["total_volume"] - volume_df["prior_volume"]) / volume_df["prior_volume"]
        ) * 100.0
        if (volume_df["volume_change_pct"] > 20.0).any():
            flags.append("Volume increase greater than 20% week-over-week")

    return flags


def process_athlete_analytics(
    nutrition_df: pd.DataFrame,
    training_df: pd.DataFrame,
    calorie_target: float | None = None,
) -> dict[str, Any]:
    """Compute deterministic nutrition/training metrics and risk flags.

    Args:
        nutrition_df: Cleaned nutrition dataframe.
        training_df: Cleaned training dataframe.
        calorie_target: Optional daily calorie target used for adherence metric.

    Returns:
        Structured metrics dictionary for downstream auditing.

    Raises:
        ProcessingError: For validation or processing failures.
    """
    try:
        nutrition_metrics = compute_weekly_nutrition_metrics(nutrition_df, calorie_target=calorie_target)
        training_metrics = compute_weekly_training_metrics(training_df)
        risk_flags = detect_risk_flags(nutrition_metrics, training_metrics)
    except ProcessingError:
        raise
    except Exception as exc:
        raise ProcessingError("Failed to process athlete analytics") from exc

    return {
        "nutrition_metrics": nutrition_metrics,
        "training_metrics": training_metrics,
        "risk_flags": risk_flags,
    }


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible placeholder used by current main pipeline wiring."""
    return records
