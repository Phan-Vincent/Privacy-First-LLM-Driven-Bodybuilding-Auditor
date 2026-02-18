"""LLM auditing module for secure model interactions and strict JSON responses."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError


logger = logging.getLogger(__name__)


class LLMAuditError(Exception):
    """Raised when LLM auditing operations fail."""


_ALLOWED_STRENGTH_TRENDS = {"Improving", "Plateau", "Declining"}

_SYSTEM_PROMPT = (
    "You are a strict analytics auditor. Return STRICT JSON only. "
    "Do not return markdown, prose, explanations, or comments. "
    "Output must exactly match this schema and field types: "
    '{"summary":"string","strength_trend":"Improving | Plateau | Declining",'
    '"nutrition_assessment":"string","risk_flags":["string"],"recommendations":["string"]}'
)


def _validate_audit_schema(payload: dict[str, Any]) -> None:
    """Validate parsed audit JSON against expected schema.

    Args:
        payload: Parsed JSON dictionary.

    Raises:
        LLMAuditError: If schema is invalid.
    """
    required_fields = {
        "summary": str,
        "strength_trend": str,
        "nutrition_assessment": str,
        "risk_flags": list,
        "recommendations": list,
    }

    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise LLMAuditError(f"Audit schema mismatch: missing fields {missing}")

    for field, expected_type in required_fields.items():
        if not isinstance(payload[field], expected_type):
            raise LLMAuditError(f"Audit schema mismatch: field '{field}' must be {expected_type.__name__}")

    if payload["strength_trend"] not in _ALLOWED_STRENGTH_TRENDS:
        raise LLMAuditError(
            "Audit schema mismatch: 'strength_trend' must be one of "
            f"{sorted(_ALLOWED_STRENGTH_TRENDS)}"
        )

    if not all(isinstance(item, str) for item in payload["risk_flags"]):
        raise LLMAuditError("Audit schema mismatch: all risk_flags entries must be strings")

    if not all(isinstance(item, str) for item in payload["recommendations"]):
        raise LLMAuditError("Audit schema mismatch: all recommendations entries must be strings")


def _safe_metrics_payload(processed_metrics: dict[str, Any]) -> str:
    """Serialize model input in a deterministic way.

    The only user-derived content sent to the model is the safely serialized
    metrics payload.

    Args:
        processed_metrics: Deterministic analytics output.

    Returns:
        JSON string for model input.

    Raises:
        LLMAuditError: If payload is not JSON serializable.
    """
    try:
        return json.dumps(processed_metrics, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise LLMAuditError("Processed metrics are not JSON serializable") from exc


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate request cost with placeholder rates.

    Rates are intentionally conservative placeholders and should be updated for
    the selected model pricing.
    """
    input_rate_per_1k = 0.005
    output_rate_per_1k = 0.015
    return round((input_tokens / 1000.0) * input_rate_per_1k + (output_tokens / 1000.0) * output_rate_per_1k, 6)


def generate_audit(processed_metrics: dict[str, Any]) -> dict[str, Any]:
    """Generate a strict JSON audit from processed deterministic metrics.

    Args:
        processed_metrics: Deterministic analytics dictionary from processing.

    Returns:
        Dictionary with parsed audit object and token usage metadata.

    Raises:
        LLMAuditError: If configuration, API call, response parsing, or schema validation fails.
    """
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise LLMAuditError("LLM_API_KEY is not configured")

    safe_payload = _safe_metrics_payload(processed_metrics)

    try:
        client = OpenAI(api_key=api_key)
        completion = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze the following processed athlete metrics and return strict JSON only: "
                        + safe_payload
                    ),
                },
            ],
            temperature=0,
        )
    except OpenAIError as exc:
        logger.exception("OpenAI request failed")
        raise LLMAuditError("Failed to generate LLM audit") from exc

    output_text = getattr(completion, "output_text", "")
    if not output_text:
        raise LLMAuditError("LLM response did not include output text")

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise LLMAuditError("LLM response is not valid JSON") from exc

    if not isinstance(parsed, dict):
        raise LLMAuditError("LLM response JSON root must be an object")

    _validate_audit_schema(parsed)

    usage = getattr(completion, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    return {
        "audit": parsed,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": _estimate_cost(input_tokens, output_tokens),
        },
    }


def call_llm_audit(payload: list[dict[str, Any]]) -> dict[str, Any]:
    """Backward-compatible wrapper for existing pipeline wiring."""
    return generate_audit({"records": payload})
