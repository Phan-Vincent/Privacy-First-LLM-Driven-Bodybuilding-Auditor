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


_ALLOWED_OVERALL_STATUS = {"On Track", "Needs Adjustment", "High Risk"}
_ALLOWED_STRENGTH_TRENDS = {"Improving", "Plateau", "Declining"}
_REQUIRED_TOP_LEVEL_KEYS = {
    "overall_status",
    "strength_trend",
    "nutrition_assessment",
    "recovery_risk_score",
    "risk_flags",
    "priority_recommendations",
}

_SYSTEM_PROMPT = (
    "You are a strict analytics auditor. Return STRICT JSON only. "
    "No markdown, no prose, and no text outside JSON. "
    "The output must exactly match this schema and field names/types: "
    '{'
    '"overall_status":"On Track | Needs Adjustment | High Risk",'
    '"strength_trend":{"status":"Improving | Plateau | Declining","velocity_percent_change":0.0},'
    '"nutrition_assessment":{"calorie_adherence_percent":0.0,"protein_per_lb":0.0,"consistency_score":0.0},'
    '"recovery_risk_score":0.0,'
    '"risk_flags":["string"],'
    '"priority_recommendations":[{"priority":1,"action":"string","reason":"string"}]'
    '}'
)


def _is_number(value: Any) -> bool:
    """Return True for non-boolean numeric values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_audit_schema(payload: dict[str, Any]) -> None:
    """Validate parsed audit JSON against expected schema."""
    missing = [field for field in _REQUIRED_TOP_LEVEL_KEYS if field not in payload]
    if missing:
        raise LLMAuditError(f"Audit schema mismatch: missing fields {missing}")

    extra = [field for field in payload if field not in _REQUIRED_TOP_LEVEL_KEYS]
    if extra:
        raise LLMAuditError(f"Audit schema mismatch: unexpected fields {extra}")

    overall_status = payload["overall_status"]
    if not isinstance(overall_status, str) or overall_status not in _ALLOWED_OVERALL_STATUS:
        raise LLMAuditError(
            "Audit schema mismatch: 'overall_status' must be one of "
            f"{sorted(_ALLOWED_OVERALL_STATUS)}"
        )

    strength_trend = payload["strength_trend"]
    if not isinstance(strength_trend, dict):
        raise LLMAuditError("Audit schema mismatch: 'strength_trend' must be an object")
    expected_strength_keys = {"status", "velocity_percent_change"}
    if set(strength_trend.keys()) != expected_strength_keys:
        raise LLMAuditError("Audit schema mismatch: 'strength_trend' keys are invalid")
    if strength_trend["status"] not in _ALLOWED_STRENGTH_TRENDS:
        raise LLMAuditError(
            "Audit schema mismatch: 'strength_trend.status' must be one of "
            f"{sorted(_ALLOWED_STRENGTH_TRENDS)}"
        )
    if not _is_number(strength_trend["velocity_percent_change"]):
        raise LLMAuditError("Audit schema mismatch: 'strength_trend.velocity_percent_change' must be a number")

    nutrition = payload["nutrition_assessment"]
    if not isinstance(nutrition, dict):
        raise LLMAuditError("Audit schema mismatch: 'nutrition_assessment' must be an object")
    expected_nutrition_keys = {"calorie_adherence_percent", "protein_per_lb", "consistency_score"}
    if set(nutrition.keys()) != expected_nutrition_keys:
        raise LLMAuditError("Audit schema mismatch: 'nutrition_assessment' keys are invalid")
    for key in expected_nutrition_keys:
        if not _is_number(nutrition[key]):
            raise LLMAuditError(f"Audit schema mismatch: 'nutrition_assessment.{key}' must be a number")

    if not _is_number(payload["recovery_risk_score"]):
        raise LLMAuditError("Audit schema mismatch: 'recovery_risk_score' must be a number")

    risk_flags = payload["risk_flags"]
    if not isinstance(risk_flags, list) or not all(isinstance(item, str) and item.strip() for item in risk_flags):
        raise LLMAuditError("Audit schema mismatch: 'risk_flags' must be a list of non-empty strings")

    recommendations = payload["priority_recommendations"]
    if not isinstance(recommendations, list):
        raise LLMAuditError("Audit schema mismatch: 'priority_recommendations' must be a list")
    for item in recommendations:
        if not isinstance(item, dict):
            raise LLMAuditError("Audit schema mismatch: each recommendation must be an object")
        expected_rec_keys = {"priority", "action", "reason"}
        if set(item.keys()) != expected_rec_keys:
            raise LLMAuditError("Audit schema mismatch: recommendation keys are invalid")
        if not isinstance(item["priority"], int):
            raise LLMAuditError("Audit schema mismatch: recommendation 'priority' must be an integer")
        if not isinstance(item["action"], str) or not item["action"].strip():
            raise LLMAuditError("Audit schema mismatch: recommendation 'action' must be a non-empty string")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise LLMAuditError("Audit schema mismatch: recommendation 'reason' must be a non-empty string")


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
    """Estimate request cost with placeholder rates."""
    input_rate_per_1k = 0.005
    output_rate_per_1k = 0.015
    return round((input_tokens / 1000.0) * input_rate_per_1k + (output_tokens / 1000.0) * output_rate_per_1k, 6)


def generate_audit(processed_metrics: dict[str, Any]) -> dict[str, Any]:
    """Generate a strict JSON audit from processed deterministic metrics."""
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise LLMAuditError("LLM_API_KEY is not configured")

    safe_payload = _safe_metrics_payload(processed_metrics)

    # SECURITY: Only sanitized structured metrics are sent to LLM.
    # Raw CSV data is never forwarded.
    try:
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
        completion = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Analyze the following processed athlete metrics and return strict JSON only: " + safe_payload,
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
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
