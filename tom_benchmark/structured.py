"""Layer 3: structured JSON output prompting & parsing."""

from __future__ import annotations

import json
import re

from .models import Layer3Result, Scenario

STRUCTURED_INSTRUCTION = """Respond ONLY with a JSON object of the form:
{"answer": "<your concise answer>", "reasoning": "<brief explanation>", "confidence": <float between 0.0 and 1.0>}

Do not include any text outside the JSON. Do not include markdown fences."""


def build_structured_prompt(scenario: Scenario) -> str:
    return (
        f"Scenario:\n{scenario.scenario}\n\n"
        f"Question: {scenario.question}\n\n"
        f"{STRUCTURED_INSTRUCTION}"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_structured_response(text: str) -> Layer3Result | None:
    """Parse a structured JSON response. Returns None if it cannot be parsed."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.MULTILINE).strip()

    match = _JSON_OBJECT_RE.search(candidate)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    answer = str(data.get("answer", "")).strip()
    reasoning = str(data.get("reasoning", "")).strip()
    raw_conf = data.get("confidence", 0.0)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if not answer:
        return None

    return Layer3Result(answer=answer, reasoning=reasoning, confidence=confidence)
