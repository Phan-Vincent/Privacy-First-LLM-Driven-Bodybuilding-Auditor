# Privacy-First LLM-Driven Bodybuilding Auditor

A secure, modular pipeline that ingests structured nutrition and strength training data, computes deterministic performance metrics, and generates a strictly validated LLM-based weekly audit.

This project demonstrates applied AI engineering, prompt injection hardening, schema validation, cost guardrails, and secure key management within a real-world personal analytics context.

It is intentionally built without unnecessary infrastructure or UI layers. The goal is clarity, security, and correctness.

---

## Overview

Modern athletes generate large volumes of structured data (nutrition logs, strength logs, bodyweight trends). Traditional rule-based analytics are useful but limited in synthesizing context.

This system:

1. Ingests structured CSV exports (Cronometer, StrengthLog).
2. Computes deterministic performance metrics.
3. Passes sanitized, structured data to an LLM.
4. Enforces strict JSON schema validation on model output.
5. Generates a weekly report.
6. Logs token usage and enforces cost guardrails.

The model is treated as an untrusted component. All outputs are validated before use.

---

## Design Principles

* Privacy first
* No raw user data sent to the model
* Strict schema enforcement
* Prompt injection mitigation
* No hardcoded secrets
* Deterministic analytics before AI
* Token cost monitoring

The LLM augments analysis — it does not replace deterministic logic.

---

## Architecture

User → CSV Input
→ Ingestion Layer (validation + sanitization)
→ Processing Layer (deterministic metrics + risk flags)
→ LLM Audit Layer (structured JSON only)
→ Schema Validation
→ Report Generator (PDF + JSON)
→ Token Usage Log

The model never receives raw CSV text. Only sanitized, computed metrics are serialized and passed into the prompt.

---

## Core Features

### Data Ingestion

* Validates file extensions
* Enforces expected schema
* Rejects malformed input
* Strips potentially unsafe content

### Deterministic Processing

* Weekly average calories and protein
* Protein per lb bodyweight
* Calorie adherence percentage
* Volume per muscle group
* Strength progression velocity
* Rule-based risk flag detection

### LLM Audit Engine

* Strict JSON-only output
* Schema validation via `jsonschema`
* Rejects unexpected or malformed fields
* Explicit prompt injection hardening
* Structured recommendation prioritization

### Security Controls

* API key via environment variables
* `.env` excluded from version control
* No secret logging
* No raw user data logged
* Strict role separation in prompts
* Token usage logging
* Cost guardrails with abort thresholds

### Reporting

* Clean weekly PDF report
* Structured JSON output for machine consumption
* Token usage summary

---

## Threat Model

### Potential Risks

* API key exposure
* Prompt injection via malformed CSV
* Schema drift from LLM output
* Data exfiltration through model hallucination
* Excessive token usage leading to cost overrun

### Mitigations

* Environment-based secret management
* No raw CSV content passed to model
* Strict JSON schema validation
* Hard failure on malformed responses
* Token budget enforcement
* Structured prompt roles
* No dynamic role modification

The LLM is treated as an untrusted data source. All outputs are validated before downstream use.

---

## Installation

Python 3.11+

```bash
git clone https://github.com/Phan-Vincent/Privacy-First-LLM-Driven-Bodybuilding-Auditor.git
cd Privacy-First-LLM-Driven-Bodybuilding-Auditor
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Usage

```bash
python main.py --nutrition cronometer.csv --training strengthlog.csv --target_calories 2800
```

Outputs:

* weekly_report.pdf
* audit.json
* token_usage.log

The CLI prints only safe summary information. No sensitive data is displayed.

---

## Example Structured Output

```json
{
  "overall_status": "Needs Adjustment",
  "strength_trend": {
    "status": "Improving",
    "velocity_percent_change": 3.4
  },
  "nutrition_assessment": {
    "calorie_adherence_percent": 92.1,
    "protein_per_lb": 0.78,
    "consistency_score": 84.5
  },
  "recovery_risk_score": 0.62,
  "risk_flags": [
    "Protein below target threshold",
    "Volume increase >20% week-over-week"
  ],
  "priority_recommendations": [
    {
      "priority": 1,
      "action": "Increase protein intake to 0.9–1.0 g/lb bodyweight",
      "reason": "Current intake below hypertrophy-optimized range"
    }
  ]
}
```

All outputs are validated against a strict schema. Unexpected fields are rejected.

---

## Token Cost Monitoring

Each run logs:

* Input tokens
* Output tokens
* Total tokens
* Estimated cost

Configurable guardrails:

* `MAX_TOKEN_LIMIT`
* `MAX_COST_PER_RUN`

If thresholds are exceeded, execution aborts safely.

---

## Testing

This project includes a pytest suite covering:

* Schema validation success and failure
* Prompt injection simulation
* Token budget enforcement
* Malformed CSV rejection
* Missing column handling

LLM calls are mocked during testing. No live API calls are made.

---

## Roadmap

* Anomaly detection layer (deterministic)
* CI integration (GitHub Actions)
* Static type checking (mypy)
* Optional encrypted local storage
* Synthetic test data generator
* Containerized reproducible environment

---

## Lessons Learned

* LLM output must be validated, never trusted.
* Structured prompts reduce hallucination surface.
* Prompt injection risk exists even in numeric datasets.
* Deterministic analytics should precede AI synthesis.
* Cost monitoring is part of secure system design.

---

## License

MIT

---
