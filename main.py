"""CLI entry point for the secure athlete analytics pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ingestion import IngestionError, SchemaMismatchError, read_csv_safely
from llm_audit import LLMAuditError, generate_audit
from processing import MissingColumnsError, ProcessingError, process_athlete_analytics
from report import ReportError, write_json_report, write_pdf_report


NUTRITION_COLUMNS: list[str] = ["date", "calories", "protein_g", "bodyweight_lb"]
TRAINING_COLUMNS: list[str] = ["date", "muscle_group", "sets", "reps", "weight"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline entry point."""
    parser = argparse.ArgumentParser(description="Run secure athlete analytics pipeline")
    parser.add_argument("--nutrition", required=True, type=Path, help="Path to cleaned nutrition CSV")
    parser.add_argument("--training", required=True, type=Path, help="Path to cleaned training CSV")
    parser.add_argument("--target_calories", required=False, type=int, help="Optional daily calorie target")
    parser.add_argument("--output_dir", required=False, type=Path, default=Path("./out"), help="Output directory")
    return parser.parse_args()


def load_input_data(nutrition_path: Path, training_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate nutrition and training CSVs through ingestion module."""
    nutrition_rows = read_csv_safely(nutrition_path, expected_columns=NUTRITION_COLUMNS)
    training_rows = read_csv_safely(training_path, expected_columns=TRAINING_COLUMNS)
    return pd.DataFrame(nutrition_rows), pd.DataFrame(training_rows)


def _configure_file_logger(output_dir: Path) -> logging.Logger:
    """Create a sanitized file logger for operational diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _log_token_usage(output_dir: Path, token_usage: dict[str, int | float]) -> None:
    """Append token usage metadata to a file in output directory with restrictive permissions."""
    usage_file = output_dir / "token_usage.log"
    serialized = json.dumps(token_usage, ensure_ascii=False)
    usage_file.parent.mkdir(parents=True, exist_ok=True)

    if not usage_file.exists():
        fd = os.open(str(usage_file), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
    with usage_file.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")


def run_cli(args: argparse.Namespace) -> int:
    """Run the pipeline orchestration from parsed CLI arguments."""
    load_dotenv()

    output_dir = args.output_dir
    logger = _configure_file_logger(output_dir)

    try:
        nutrition_df, training_df = load_input_data(args.nutrition, args.training)
        processed_metrics = process_athlete_analytics(
            nutrition_df=nutrition_df,
            training_df=training_df,
            calorie_target=args.target_calories,
        )
        audit_result = generate_audit(processed_metrics)

        json_path = output_dir / "audit_report.json"
        pdf_path = output_dir / "audit_report.pdf"

        full_report = {
            "processed_metrics": processed_metrics,
            "audit": audit_result["audit"],
            "token_usage": audit_result["token_usage"],
        }
        write_json_report(full_report, json_path)
        write_pdf_report(full_report, pdf_path)
        _log_token_usage(output_dir, audit_result["token_usage"])

        logger.info("Pipeline completed successfully")
        print(f"Pipeline completed successfully. Reports written to: {output_dir}")
        return 0

    except (IngestionError, SchemaMismatchError, MissingColumnsError) as exc:
        logger.warning("Input validation failed: %s", type(exc).__name__)
        print("Error: Input validation failed. Please verify CSV files and required columns.")
        return 1
    except ProcessingError as exc:
        logger.warning("Metrics processing failed: %s", type(exc).__name__)
        print("Error: Metrics processing failed. Please verify input data formats.")
        return 1
    except LLMAuditError as exc:
        logger.warning("Audit generation failed: %s", type(exc).__name__)
        print("Error: Audit generation failed. Please verify environment and model configuration.")
        return 1
    except ReportError as exc:
        logger.warning("Report generation failed: %s", type(exc).__name__)
        print("Error: Report generation failed. Please check output path permissions.")
        return 1
    except Exception as exc:
        logger.exception("Unexpected pipeline failure: %s", type(exc).__name__)
        print("Error: Unexpected pipeline failure.")
        return 1


def main() -> int:
    """Program entrypoint."""
    args = parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
