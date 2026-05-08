from tom_benchmark.scorer import extract_answer, score_response


def test_direct_answer_correct(sally_anne):
    out = score_response(sally_anne, "basket")
    assert out.correct is True
    assert out.ambiguous is False


def test_alias_match(sally_anne):
    out = score_response(sally_anne, "She will look in the basket first.")
    assert out.correct is True


def test_wrong_short_answer(sally_anne):
    out = score_response(sally_anne, "box")
    assert out.correct is False
    assert out.ambiguous is False


def test_negation_avoids_false_positive(sally_anne):
    out = score_response(sally_anne, "She will not look in the basket; she'll look in the box.")
    assert out.correct is False or out.correct is None  # either incorrect or ambiguous


def test_yes_no_ambiguous_when_both_appear(yes_no_scenario):
    out = score_response(
        yes_no_scenario,
        "Some might say yes, but the better answer is no — Tom didn't hear the broadcast and so cannot know.",
    )
    # Both 'yes' and 'no' show up — Layer 1 should defer.
    assert out.correct in {None, True}  # ambiguous-or-correct, not False


def test_extract_answer_lead_in():
    text = "After thinking about it, the answer is the basket."
    extracted = extract_answer(text)
    assert extracted is not None
    assert "basket" in extracted.lower()


def test_extract_answer_will_look():
    text = "Sally will look in the basket because she didn't see Anne move it."
    extracted = extract_answer(text)
    assert extracted is not None
    assert "basket" in extracted.lower()


def test_empty_response(sally_anne):
    out = score_response(sally_anne, "")
    assert out.correct is False


def test_verbose_no_match_is_ambiguous(sally_anne):
    response = (
        "Considering the cognitive states involved, perspective-taking, "
        "and the fact that one party was absent during the relevant event, "
        "this is a classic example of belief-reality divergence. "
        "I would say the relevant container that initially held the object."
    )
    out = score_response(sally_anne, response)
    # Should be flagged ambiguous since 'basket' isn't actually in the text.
    assert out.ambiguous is True
    assert out.correct is None
