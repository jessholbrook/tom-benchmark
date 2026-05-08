"""Click-based CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click
from dotenv import load_dotenv

from .exporter import export as export_dataset
from .judge import JudgeConfig
from .loader import CATEGORY_FILES, category_counts, load_scenarios
from .models import BenchmarkRun
from .runner import BenchmarkRunner

load_dotenv()


CATEGORY_CHOICES = list(CATEGORY_FILES.keys())
TIER_CHOICES = ["easy", "medium", "hard", "expert"]


@click.group()
@click.version_option(package_name="tom-benchmark")
def cli() -> None:
    """Theory of Mind benchmark CLI."""


@cli.command("list-categories")
def list_categories_cmd() -> None:
    """List the 6 ToM categories with their scenario counts."""
    counts = category_counts()
    click.echo(f"{'CATEGORY':<24} COUNT")
    click.echo("-" * 32)
    total = 0
    for cat, count in counts.items():
        click.echo(f"{cat:<24} {count}")
        total += count
    click.echo("-" * 32)
    click.echo(f"{'TOTAL':<24} {total}")


@cli.command("list-scenarios")
@click.option("--category", type=click.Choice(CATEGORY_CHOICES), default=None)
@click.option("--tier", type=click.Choice(TIER_CHOICES), default=None)
def list_scenarios_cmd(category: str | None, tier: str | None) -> None:
    """List individual scenarios with optional filters."""
    scenarios = load_scenarios(category=category, tier=tier)
    if not scenarios:
        click.echo("No scenarios match the given filters.")
        return
    click.echo(f"{'ID':<10} {'CATEGORY':<22} {'TIER':<8} NAME")
    click.echo("-" * 70)
    for s in scenarios:
        click.echo(f"{s.id:<10} {s.category:<22} {s.tier:<8} {s.name}")
    click.echo(f"\n{len(scenarios)} scenarios.")


@cli.command("run")
@click.option(
    "--models",
    "models",
    multiple=True,
    required=True,
    help="Model name(s) to evaluate (repeatable).",
)
@click.option("--category", type=click.Choice(CATEGORY_CHOICES), default=None)
@click.option("--tier", type=click.Choice(TIER_CHOICES), default=None)
@click.option("--output", type=click.Path(dir_okay=False), default=None, help="Save results to JSON file.")
@click.option("--judge", "judge_model", default=None, help="Enable Layer 2 with this judge model.")
@click.option(
    "--judge-always",
    is_flag=True,
    default=False,
    help="Run the judge on every response, not just ambiguous ones.",
)
@click.option("--structured", is_flag=True, default=False, help="Enable Layer 3 structured output.")
def run_cmd(
    models: tuple[str, ...],
    category: str | None,
    tier: str | None,
    output: str | None,
    judge_model: str | None,
    judge_always: bool,
    structured: bool,
) -> None:
    """Run the benchmark against one or more models."""
    scenarios = load_scenarios(category=category, tier=tier)
    if not scenarios:
        click.echo("No scenarios match the given filters.", err=True)
        raise click.Abort()

    judge_cfg = JudgeConfig(judge_model=judge_model, always=judge_always) if judge_model else None

    runs: list[BenchmarkRun] = []
    for model in models:
        click.echo(f"\n>>> Running {model} on {len(scenarios)} scenarios...")
        runner = BenchmarkRunner(
            model_name=model,
            judge=judge_cfg,
            structured=structured,
        )

        def _progress(idx: int, total: int, scen) -> None:
            click.echo(f"  [{idx}/{total}] {scen.id} ({scen.category}/{scen.tier})")

        run = runner.run(scenarios, progress=_progress)
        runs.append(run)
        click.echo(
            f"<<< {model}: accuracy = {run.accuracy:.2%} ({run.correct_count}/{run.total})"
        )

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": [json.loads(r.model_dump_json()) for r in runs]}
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        click.echo(f"\nResults written to {out_path}")


@cli.command("results")
@click.argument("results_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def results_cmd(results_file: str, fmt: str) -> None:
    """Display results from a previous run file."""
    payload = json.loads(Path(results_file).read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if fmt == "json":
        click.echo(json.dumps(runs, indent=2, default=str))
        return

    for run in runs:
        click.echo(f"\nMODEL: {run['model_name']}  ({run['timestamp']})")
        click.echo(f"  total: {run['total']}, correct: {run['correct_count']}, accuracy: {run['accuracy']:.2%}")

        by_cat: dict[str, list[bool]] = {}
        by_tier: dict[str, list[bool]] = {}
        for r in run["results"]:
            if r.get("correct") is None:
                continue
            by_cat.setdefault(r["category"], []).append(bool(r["correct"]))
            by_tier.setdefault(r["tier"], []).append(bool(r["correct"]))

        click.echo("  by category:")
        for cat, vals in sorted(by_cat.items()):
            click.echo(f"    {cat:<24} {sum(vals) / len(vals):.2%}  ({sum(vals)}/{len(vals)})")
        click.echo("  by tier:")
        for tier, vals in sorted(by_tier.items()):
            click.echo(f"    {tier:<10} {sum(vals) / len(vals):.2%}  ({sum(vals)}/{len(vals)})")


@cli.command("export")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "csv", "huggingface"]),
    required=True,
)
@click.option("--output", type=click.Path(), required=True)
@click.option("--category", type=click.Choice(CATEGORY_CHOICES), default=None)
@click.option("--tier", type=click.Choice(TIER_CHOICES), default=None)
def export_cmd(fmt: str, output: str, category: str | None, tier: str | None) -> None:
    """Export the scenario dataset."""
    scenarios = load_scenarios(category=category, tier=tier)
    if not scenarios:
        click.echo("No scenarios match the given filters.", err=True)
        raise click.Abort()
    path = export_dataset(scenarios, fmt, output)
    click.echo(f"Exported {len(scenarios)} scenarios to {path}")


if __name__ == "__main__":
    cli()
