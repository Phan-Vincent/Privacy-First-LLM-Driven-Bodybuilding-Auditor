"""Entry point for the local AI analytics pipeline skeleton."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ingestion import IngestionError, read_csv_safely
from llm_audit import LLMAuditError, call_llm_audit
from processing import ProcessingError, normalize_records
from report import ReportError, write_json_report


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic application logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def run_pipeline(input_csv: Path, output_json: Path) -> None:
    """Execute the pipeline using safe defaults and modular components.

    Args:
        input_csv: Input CSV path.
        output_json: Output JSON report path.
    """
    expected_columns = ["athlete", "date", "exercise", "reps", "weight"]
    records = read_csv_safely(input_csv, expected_columns=expected_columns)
    processed = normalize_records(records)
    result = call_llm_audit(processed)
    write_json_report(result, output_json)


def main() -> int:
    """CLI-style main function with error-handling scaffolding.

    Returns:
        Process exit code.
    """
    configure_logging()
    load_dotenv()

    input_csv = Path(os.getenv("INPUT_CSV_PATH", "./data/input.csv"))
    output_json = Path(os.getenv("OUTPUT_REPORT_PATH", "./out/audit_report.json"))

    try:
        run_pipeline(input_csv=input_csv, output_json=output_json)
    except IngestionError:
        logger.exception("Ingestion stage failed")
        return 1
    except ProcessingError:
        logger.exception("Processing stage failed")
        return 1
    except LLMAuditError:
        logger.exception("LLM audit stage failed")
        return 1
    except ReportError:
        logger.exception("Report stage failed")
        return 1

    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
