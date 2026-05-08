"""Shared fixtures and helpers for tests."""

from __future__ import annotations

import pytest

from tom_benchmark.adapters.base import LLMAdapter, LLMResponse
from tom_benchmark.models import Complexity, Scenario


class FakeAdapter(LLMAdapter):
    """A scripted adapter for tests — returns canned responses by index."""

    def __init__(self, responses: list[str], model: str = "fake-model") -> None:
        super().__init__(model)
        self.responses = list(responses)
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def query(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        idx = len(self.calls)
        if idx >= len(self.responses):
            text = self.responses[-1]
        else:
            text = self.responses[idx]
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return LLMResponse(text=text, model=self.model, latency_seconds=0.001)


@pytest.fixture
def sally_anne() -> Scenario:
    return Scenario(
        id="fb_001",
        category="false_belief",
        name="Sally-Anne",
        tier="easy",
        complexity=Complexity(belief_depth=1, num_agents=2, information_gap_type="absence"),
        scenario="Sally puts marble in basket; Anne moves it to box while Sally is away.",
        question="Where will Sally look first?",
        expected_answer="basket",
        answer_aliases=["the basket", "in the basket"],
        rubric="Must say basket.",
        explanation="Sally didn't see the move.",
    )


@pytest.fixture
def yes_no_scenario() -> Scenario:
    return Scenario(
        id="ka_001",
        category="knowledge_attr",
        name="Yes/No",
        tier="easy",
        complexity=Complexity(belief_depth=1, num_agents=1, information_gap_type="absence"),
        scenario="Tom is in the basement; the radio is in the kitchen.",
        question="Does Tom know the announcement?",
        expected_answer="no",
        answer_aliases=["he does not know", "no he does not"],
        rubric="Must say no.",
        explanation="Tom didn't hear it.",
    )
