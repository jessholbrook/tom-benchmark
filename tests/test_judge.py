from tom_benchmark.judge import JudgeScorer, _parse_judge_output


def test_parse_judge_output_correct():
    raw = '{"score": 0.95, "verdict": "correct", "rationale": "Right answer."}'
    result = _parse_judge_output(raw, judge_model="judge-x")
    assert result.score == 0.95
    assert result.verdict == "correct"
    assert result.judge_model == "judge-x"


def test_parse_judge_output_clamps_score():
    raw = '{"score": 1.5, "verdict": "correct", "rationale": "x"}'
    result = _parse_judge_output(raw, judge_model="j")
    assert result.score == 1.0


def test_parse_judge_output_handles_markdown_fences():
    raw = '```json\n{"score": 0.5, "verdict": "partial", "rationale": "Mixed."}\n```'
    result = _parse_judge_output(raw, judge_model="j")
    assert result.verdict == "partial"
    assert result.score == 0.5


def test_parse_judge_output_invalid_verdict_normalizes():
    raw = '{"score": 0.0, "verdict": "garbage", "rationale": "x"}'
    result = _parse_judge_output(raw, judge_model="j")
    assert result.verdict == "incorrect"


def test_parse_judge_output_no_json():
    result = _parse_judge_output("no JSON here", judge_model="j")
    assert result.verdict == "incorrect"
    assert result.score == 0.0


def test_judge_scorer_uses_adapter(sally_anne):
    from tests.conftest import FakeAdapter

    adapter = FakeAdapter(['{"score": 1.0, "verdict": "correct", "rationale": "ok"}'])
    scorer = JudgeScorer(judge_model="fake-judge", adapter=adapter)
    result = scorer.evaluate(sally_anne, "She'll look in the basket.")
    assert result.verdict == "correct"
    assert result.score == 1.0
    assert len(adapter.calls) == 1
    # System prompt should be present.
    assert adapter.calls[0]["system_prompt"]
