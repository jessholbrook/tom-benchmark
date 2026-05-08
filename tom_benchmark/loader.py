"""Load scenarios from the bundled JSON files, with optional category/tier filtering."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from .models import Category, Scenario, Tier

CATEGORY_FILES: dict[str, str] = {
    "false_belief": "false_belief.json",
    "indirect_request": "indirect_request.json",
    "knowledge_attr": "knowledge_attr.json",
    "deception": "deception.json",
    "emotional_inference": "emotional_inference.json",
    "higher_order_beliefs": "higher_order_beliefs.json",
}


def _scenarios_dir() -> Path:
    return Path(str(files("tom_benchmark").joinpath("scenarios")))


def _read_category_file(category: str) -> list[Scenario]:
    filename = CATEGORY_FILES.get(category)
    if filename is None:
        raise ValueError(f"Unknown category: {category!r}")
    path = _scenarios_dir() / filename
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Scenario.model_validate(s) for s in raw.get("scenarios", [])]


def load_scenarios(
    category: Category | str | None = None,
    tier: Tier | str | None = None,
) -> list[Scenario]:
    """Load scenarios, optionally filtered by category and/or difficulty tier."""
    if category is not None:
        scenarios = _read_category_file(str(category))
    else:
        scenarios = []
        for cat in CATEGORY_FILES:
            scenarios.extend(_read_category_file(cat))

    if tier is not None:
        scenarios = [s for s in scenarios if s.tier == tier]
    return scenarios


def category_counts() -> dict[str, int]:
    """Return a mapping of category name to scenario count."""
    return {cat: len(_read_category_file(cat)) for cat in CATEGORY_FILES}


def list_categories() -> list[str]:
    return list(CATEGORY_FILES.keys())
