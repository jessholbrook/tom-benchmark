"""Layer 2: LLM-as-Judge scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .adapters import LLMAdapter, get_adapter
from .models import Layer2Result, Scenario

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of Theory of Mind responses.

You will be given a scenario, the question asked, the expected answer, a grading rubric, and a candidate model's response. Your job is to grade the response against the rubric.

Respond with ONLY a JSON object of the form:
{"score": <float between 0.0 and 1.0>, "verdict": "<correct|partial|incorrect>", "rationale": "<one or two sentences>"}

Do not include any text outside the JSON object. Do not include markdown fences.
"""

JUDGE_USER_TEMPLATE = """SCENARIO:
{scenario}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

RUBRIC:
{rubric}

CANDIDATE RESPONSE:
{response}
"""


@dataclass
class JudgeConfig:
    judge_model: str
    always: bool = False
    categories: set[str] | None = None  # If set, only judge these categories.


class JudgeScorer:
    def __init__(self, judge_model: str, adapter: LLMAdapter | None = None) -> None:
        self.judge_model = judge_model
        self.adapter = adapter or get_adapter(judge_model)

    def evaluate(self, scenario: Scenario, response: str) -> Layer2Result:
        user = JUDGE_USER_TEMPLATE.format(
            scenario=scenario.scenario,
            question=scenario.question,
            expected_answer=scenario.expected_answer,
            rubric=scenario.rubric,
            response=response,
        )
        result = self.adapter.query(prompt=user, system_prompt=JUDGE_SYSTEM_PROMPT)
        return _parse_judge_output(result.text, judge_model=self.judge_model)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(text: str, judge_model: str) -> Layer2Result:
    if not text:
        return Layer2Result(score=0.0, verdict="incorrect", rationale="Empty judge response.", judge_model=judge_model)

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.MULTILINE).strip()

    match = _JSON_OBJECT_RE.search(candidate)
    if not match:
        return Layer2Result(
            score=0.0,
            verdict="incorrect",
            rationale=f"Could not parse judge output: {text[:120]}",
            judge_model=judge_model,
        )

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return Layer2Result(
            score=0.0,
            verdict="incorrect",
            rationale=f"Invalid JSON from judge: {e}",
            judge_model=judge_model,
        )

    raw_score = data.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    verdict = str(data.get("verdict", "incorrect")).strip().lower()
    if verdict not in {"correct", "partial", "incorrect"}:
        verdict = "incorrect"
    rationale = str(data.get("rationale", "")).strip()
    return Layer2Result(score=score, verdict=verdict, rationale=rationale, judge_model=judge_model)
