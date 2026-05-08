"""Benchmark orchestration engine."""

from __future__ import annotations

from typing import Callable, Iterable

from .adapters import LLMAdapter, get_adapter
from .judge import JudgeConfig, JudgeScorer
from .models import BenchmarkRun, Scenario, TestResult
from .scorer import score_response
from .structured import build_structured_prompt, parse_structured_response

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful reasoner. Read the scenario, think step by step about the "
    "mental states of the characters involved, and answer the question concisely."
)


class BenchmarkRunner:
    """Runs a list of scenarios against a single model."""

    def __init__(
        self,
        model_name: str,
        adapter: LLMAdapter | None = None,
        judge: JudgeConfig | None = None,
        judge_scorer: JudgeScorer | None = None,
        structured: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.adapter = adapter or get_adapter(model_name)
        self.judge_cfg = judge
        self._judge_scorer: JudgeScorer | None = judge_scorer
        if judge is not None and self._judge_scorer is None:
            self._judge_scorer = JudgeScorer(judge.judge_model)
        self.structured = structured
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def run(
        self,
        scenarios: Iterable[Scenario],
        progress: Callable[[int, int, Scenario], None] | None = None,
    ) -> BenchmarkRun:
        run = BenchmarkRun(
            model_name=self.model_name,
            judge_model=self.judge_cfg.judge_model if self.judge_cfg else None,
            structured=self.structured,
        )
        scenarios_list = list(scenarios)
        total = len(scenarios_list)
        for idx, scenario in enumerate(scenarios_list, start=1):
            if progress is not None:
                progress(idx, total, scenario)
            result = self._run_one(scenario)
            run.results.append(result)
        return run

    def _run_one(self, scenario: Scenario) -> TestResult:
        if self.structured:
            prompt = build_structured_prompt(scenario)
        else:
            prompt = (
                f"Scenario:\n{scenario.scenario}\n\n"
                f"Question: {scenario.question}\n\n"
                "Provide a clear, direct answer."
            )

        try:
            llm_resp = self.adapter.query(prompt=prompt, system_prompt=self.system_prompt)
        except Exception as e:  # noqa: BLE001
            return TestResult(
                scenario_id=scenario.id,
                category=scenario.category,
                tier=scenario.tier,
                model=self.model_name,
                response="",
                correct=False,
                error=str(e),
            )

        layer3 = None
        scoring_text = llm_resp.text
        if self.structured:
            layer3 = parse_structured_response(llm_resp.text)
            if layer3 is not None:
                scoring_text = layer3.answer

        outcome = score_response(scenario, scoring_text)

        layer2 = None
        should_judge = (
            self._judge_scorer is not None
            and (self.judge_cfg.always or outcome.ambiguous)  # type: ignore[union-attr]
            and self._judge_filters_allow(scenario)
        )
        if should_judge:
            try:
                layer2 = self._judge_scorer.evaluate(scenario, llm_resp.text)  # type: ignore[union-attr]
            except Exception as e:  # noqa: BLE001
                layer2 = None
                judge_error = f"judge error: {e}"
            else:
                judge_error = None
        else:
            judge_error = None

        correct = outcome.correct
        if correct is None and layer2 is not None:
            correct = layer2.verdict == "correct" or layer2.score >= 0.75

        return TestResult(
            scenario_id=scenario.id,
            category=scenario.category,
            tier=scenario.tier,
            model=self.model_name,
            response=llm_resp.text,
            extracted_answer=outcome.extracted_answer,
            correct=correct,
            ambiguous=outcome.ambiguous and layer2 is None,
            latency_seconds=llm_resp.latency_seconds,
            error=judge_error,
            layer2=layer2,
            layer3=layer3,
        )

    def _judge_filters_allow(self, scenario: Scenario) -> bool:
        cfg = self.judge_cfg
        if cfg is None or cfg.categories is None:
            return True
        return scenario.category in cfg.categories
