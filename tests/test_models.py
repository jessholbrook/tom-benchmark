from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tom_benchmark.models import (
    BenchmarkRun,
    Complexity,
    Layer2Result,
    Layer3Result,
    Scenario,
    TestResult,
)


def test_complexity_validates_belief_depth():
    with pytest.raises(ValidationError):
        Complexity(belief_depth=0, num_agents=1)
    with pytest.raises(ValidationError):
        Complexity(belief_depth=6, num_agents=1)


def test_scenario_round_trip():
    scenario = Scenario(
        id="x_001",
        category="false_belief",
        name="x",
        tier="easy",
        complexity=Complexity(belief_depth=1, num_agents=2),
        scenario="...",
        question="...",
        expected_answer="basket",
        rubric="...",
        explanation="...",
    )
    dumped = scenario.model_dump()
    restored = Scenario.model_validate(dumped)
    assert restored == scenario


def test_layer2_score_clamping_via_validation():
    with pytest.raises(ValidationError):
        Layer2Result(score=1.5, verdict="correct", rationale="...", judge_model="m")
    with pytest.raises(ValidationError):
        Layer3Result(answer="x", reasoning="y", confidence=2.0)


def test_benchmark_run_accuracy(sally_anne):
    run = BenchmarkRun(model_name="m", timestamp=datetime.now(timezone.utc))
    run.results.append(
        TestResult(
            scenario_id=sally_anne.id,
            category=sally_anne.category,
            tier=sally_anne.tier,
            model="m",
            response="basket",
            correct=True,
        )
    )
    run.results.append(
        TestResult(
            scenario_id=sally_anne.id,
            category=sally_anne.category,
            tier=sally_anne.tier,
            model="m",
            response="box",
            correct=False,
        )
    )
    run.results.append(
        TestResult(
            scenario_id=sally_anne.id,
            category=sally_anne.category,
            tier=sally_anne.tier,
            model="m",
            response="...",
            correct=None,
            ambiguous=True,
        )
    )
    assert run.total == 3
    assert run.correct_count == 1
    # Ambiguous results are excluded from accuracy denominator.
    assert run.accuracy == pytest.approx(0.5)


def test_benchmark_run_breakdowns(sally_anne):
    run = BenchmarkRun(model_name="m")
    run.results.append(
        TestResult(
            scenario_id="a",
            category="false_belief",
            tier="easy",
            model="m",
            response="basket",
            correct=True,
        )
    )
    run.results.append(
        TestResult(
            scenario_id="b",
            category="false_belief",
            tier="hard",
            model="m",
            response="box",
            correct=False,
        )
    )
    by_cat = run.accuracy_by_category()
    by_tier = run.accuracy_by_tier()
    assert by_cat["false_belief"] == pytest.approx(0.5)
    assert by_tier["easy"] == pytest.approx(1.0)
    assert by_tier["hard"] == pytest.approx(0.0)
