"""Schema validation across every bundled scenario."""

from tom_benchmark.loader import CATEGORY_FILES, load_scenarios


def test_every_scenario_loads_and_validates():
    scenarios = load_scenarios()
    assert len(scenarios) >= len(CATEGORY_FILES)  # at least one per category


def test_ids_are_unique():
    scenarios = load_scenarios()
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids)), "Duplicate scenario IDs detected"


def test_required_fields_present():
    for s in load_scenarios():
        assert s.scenario.strip()
        assert s.question.strip()
        assert s.expected_answer.strip()
        assert s.rubric.strip()
        assert s.explanation.strip()
        assert s.complexity.belief_depth >= 1
        assert s.complexity.num_agents >= 1


def test_categories_match_filenames():
    for s in load_scenarios():
        assert s.category in CATEGORY_FILES


def test_tiers_are_valid():
    valid = {"easy", "medium", "hard", "expert"}
    for s in load_scenarios():
        assert s.tier in valid


def test_expected_answer_is_not_in_aliases_redundantly():
    for s in load_scenarios():
        # Aliases shouldn't exactly equal the expected answer.
        assert s.expected_answer not in s.answer_aliases, (
            f"Scenario {s.id} has expected_answer duplicated in answer_aliases"
        )
