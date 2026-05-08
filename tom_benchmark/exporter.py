"""Dataset export to JSONL, CSV, and HuggingFace formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import Scenario


def export_jsonl(scenarios: list[Scenario], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(s.model_dump_json() + "\n")
    return output_path


def export_csv(scenarios: list[Scenario], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "name",
        "tier",
        "complexity",
        "scenario",
        "question",
        "expected_answer",
        "answer_aliases",
        "rubric",
        "explanation",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scenarios:
            row = s.model_dump()
            row["complexity"] = json.dumps(row["complexity"], ensure_ascii=False)
            row["answer_aliases"] = json.dumps(row["answer_aliases"], ensure_ascii=False)
            writer.writerow(row)
    return output_path


def export_huggingface(scenarios: list[Scenario], output_dir: Path) -> Path:
    """Write a folder with `data.jsonl` and `dataset_info.json`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.jsonl"
    export_jsonl(scenarios, data_path)

    info = {
        "description": "Theory of Mind benchmark scenarios.",
        "license": "MIT",
        "version": "1.0.0",
        "num_examples": len(scenarios),
        "features": {
            "id": "string",
            "category": "string",
            "name": "string",
            "tier": "string",
            "complexity": "object",
            "scenario": "string",
            "question": "string",
            "expected_answer": "string",
            "answer_aliases": "list<string>",
            "rubric": "string",
            "explanation": "string",
        },
        "splits": {"train": {"num_examples": len(scenarios)}},
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def export(scenarios: list[Scenario], format: str, output_path: Path | str) -> Path:
    """Dispatch to the appropriate exporter."""
    fmt = format.lower()
    output_path = Path(output_path)
    if fmt == "jsonl":
        return export_jsonl(scenarios, output_path)
    if fmt == "csv":
        return export_csv(scenarios, output_path)
    if fmt in {"huggingface", "hf"}:
        return export_huggingface(scenarios, output_path)
    raise ValueError(f"Unknown export format: {format!r}")
