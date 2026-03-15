"""
llm.py - OpenAI LLM reasoning for medical analysis.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI  # type: ignore

from config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Medical reasoning
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a highly experienced medical AI assistant trained to analyse medical reports.
Your role is to:
1. Identify abnormal medical parameters from the extracted text.
2. Detect possible medical conditions based on those parameters.
3. Classify severity as Mild, Moderate, or Emergency.
4. Recommend the appropriate doctor specialization.
5. Provide clear, simple health guidance and precautions.

Always respond ONLY with a valid JSON object matching the schema below.
Do NOT include markdown fences or any extra text.

JSON Schema:
{
  "extracted_parameters": [{"parameter": "...", "value": "...", "normal_range": "...", "status": "Normal|Abnormal|Critical"}],
  "detected_conditions": [{"condition": "...", "confidence": "High|Medium|Low", "explanation": "..."}],
  "severity": "Mild|Moderate|Emergency",
  "severity_reason": "...",
  "doctor_specialization": "...",
  "precautions": ["..."],
  "health_guidance": "...",
  "summary": "..."
}"""


def analyse_medical_report(
    report_text: str,
    rag_context: list[dict],
    patient_info: dict | None = None,
) -> dict[str, Any]:
    """
    Send report text + RAG context to GPT-4o for medical reasoning.

    Returns structured JSON dict with conditions, severity, guidance.
    """
    # Build context string from RAG chunks
    rag_string = "\n---\n".join(
        f"[{c['category'].upper()}] {c['text']}" for c in rag_context
    ) or "No additional context available."

    user_message = f"""## Medical Report Text
{report_text[:8000]}

## Retrieved Medical Knowledge
{rag_string[:4000]}

{"## Patient Information\n" + json.dumps(patient_info) if patient_info else ""}

Analyse the medical report above and return a structured JSON response."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
        logger.info(
            "LLM analysis complete. Conditions: %s | Severity: %s",
            [c.get("condition") for c in result.get("detected_conditions", [])],
            result.get("severity"),
        )
        return result

    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON: %s", exc)
        return _fallback_response("LLM returned malformed JSON.")
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        return _fallback_response(str(exc))


def _fallback_response(reason: str) -> dict:
    return {
        "extracted_parameters": [],
        "detected_conditions": [],
        "severity": "Unknown",
        "severity_reason": reason,
        "doctor_specialization": "General Practitioner",
        "precautions": ["Please consult a doctor for proper evaluation."],
        "health_guidance": "Unable to analyse report automatically. Please seek professional medical advice.",
        "summary": "Analysis could not be completed.",
        "error": reason,
    }
