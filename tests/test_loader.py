import pytest

from tom_benchmark.loader import (
    CATEGORY_FILES,
    category_counts,
    list_categories,
    load_scenarios,
)


def test_list_categories_has_all_six():
    cats = list_categories()
    assert set(cats) == set(CATEGORY_FILES.keys())
    assert len(cats) == 6


def test_load_all_scenarios_at_least_seed():
    scenarios = load_scenarios()
    assert len(scenarios) > 0
    # Seed should have at least one per category.
    cats = {s.category for s in scenarios}
    assert cats == set(CATEGORY_FILES.keys())


def test_load_filters_by_category():
    scenarios = load_scenarios(category="false_belief")
    assert all(s.category == "false_belief" for s in scenarios)


def test_load_filters_by_tier():
    scenarios = load_scenarios(tier="easy")
    assert all(s.tier == "easy" for s in scenarios)


def test_load_filters_combined():
    scenarios = load_scenarios(category="false_belief", tier="easy")
    for s in scenarios:
        assert s.category == "false_belief"
        assert s.tier == "easy"


def test_category_counts_matches_loader():
    counts = category_counts()
    for cat, expected in counts.items():
        actual = len(load_scenarios(category=cat))
        assert actual == expected


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        load_scenarios(category="bogus")
