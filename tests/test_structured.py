from tom_benchmark.structured import build_structured_prompt, parse_structured_response


def test_parse_valid_response():
    raw = '{"answer": "basket", "reasoning": "She didn\'t see the move.", "confidence": 0.95}'
    result = parse_structured_response(raw)
    assert result is not None
    assert result.answer == "basket"
    assert result.confidence == 0.95


def test_parse_clamps_confidence():
    raw = '{"answer": "basket", "reasoning": "x", "confidence": 5}'
    result = parse_structured_response(raw)
    assert result is not None
    assert result.confidence == 1.0


def test_parse_handles_markdown():
    raw = '```json\n{"answer": "basket", "reasoning": "x", "confidence": 0.5}\n```'
    result = parse_structured_response(raw)
    assert result is not None
    assert result.answer == "basket"


def test_parse_returns_none_on_empty_answer():
    raw = '{"answer": "", "reasoning": "x", "confidence": 0.9}'
    assert parse_structured_response(raw) is None


def test_parse_returns_none_on_garbage():
    assert parse_structured_response("hello world") is None


def test_build_prompt_contains_scenario_and_question(sally_anne):
    prompt = build_structured_prompt(sally_anne)
    assert sally_anne.scenario in prompt
    assert sally_anne.question in prompt
    assert "JSON" in prompt
