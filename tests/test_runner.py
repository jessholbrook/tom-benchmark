from tom_benchmark.judge import JudgeConfig, JudgeScorer
from tom_benchmark.runner import BenchmarkRunner

from tests.conftest import FakeAdapter


def test_run_single_scenario_correct(sally_anne):
    adapter = FakeAdapter(["She will look in the basket."])
    runner = BenchmarkRunner("fake-model", adapter=adapter)
    run = runner.run([sally_anne])
    assert run.total == 1
    assert run.correct_count == 1
    assert run.results[0].correct is True


def test_run_records_error_on_adapter_exception(sally_anne):
    class BoomAdapter(FakeAdapter):
        def query(self, prompt, system_prompt=None):
            raise RuntimeError("api down")

    runner = BenchmarkRunner("fake-model", adapter=BoomAdapter([""]))
    run = runner.run([sally_anne])
    assert run.results[0].correct is False
    assert run.results[0].error == "api down"


def test_run_with_structured_layer3(sally_anne):
    adapter = FakeAdapter(['{"answer": "basket", "reasoning": "didn\'t see the move", "confidence": 0.9}'])
    runner = BenchmarkRunner("fake-model", adapter=adapter, structured=True)
    run = runner.run([sally_anne])
    result = run.results[0]
    assert result.layer3 is not None
    assert result.layer3.answer == "basket"
    assert result.layer3.confidence == 0.9
    assert result.correct is True


def test_run_invokes_judge_when_ambiguous(sally_anne):
    # Adapter returns verbose answer that doesn't contain 'basket' — Layer 1 ambiguous.
    model_adapter = FakeAdapter(
        [
            "Considering perspective-taking and information access, the relevant target "
            "is the original container she placed it in originally before stepping out, "
            "namely the receptacle on the left side of the room near the window, etc."
        ]
    )

    # Judge adapter returns a 'correct' verdict.
    judge_adapter = FakeAdapter(['{"score": 0.95, "verdict": "correct", "rationale": "implied basket"}'])
    judge_scorer = JudgeScorer(judge_model="fake-judge", adapter=judge_adapter)

    runner = BenchmarkRunner(
        "fake-model",
        adapter=model_adapter,
        judge=JudgeConfig(judge_model="fake-judge", always=False),
        judge_scorer=judge_scorer,
    )

    run = runner.run([sally_anne])
    result = run.results[0]
    assert result.layer2 is not None
    assert result.correct is True  # promoted by judge
    assert result.ambiguous is False


def test_run_judge_always(sally_anne):
    model_adapter = FakeAdapter(["basket"])  # Layer 1 = correct
    judge_adapter = FakeAdapter(['{"score": 1.0, "verdict": "correct", "rationale": "ok"}'])
    judge_scorer = JudgeScorer(judge_model="fake-judge", adapter=judge_adapter)

    runner = BenchmarkRunner(
        "fake-model",
        adapter=model_adapter,
        judge=JudgeConfig(judge_model="fake-judge", always=True),
        judge_scorer=judge_scorer,
    )

    run = runner.run([sally_anne])
    # Even though Layer 1 was correct, judge should also have run.
    assert run.results[0].layer2 is not None
