import json

from click.testing import CliRunner

from tom_benchmark.cli import cli


def test_list_categories_lists_all_six():
    runner = CliRunner()
    result = runner.invoke(cli, ["list-categories"])
    assert result.exit_code == 0
    for cat in [
        "false_belief",
        "indirect_request",
        "knowledge_attr",
        "deception",
        "emotional_inference",
        "higher_order_beliefs",
    ]:
        assert cat in result.output


def test_list_scenarios_with_filter():
    runner = CliRunner()
    result = runner.invoke(cli, ["list-scenarios", "--category", "false_belief"])
    assert result.exit_code == 0
    assert "fb_001" in result.output


def test_list_scenarios_tier_filter():
    runner = CliRunner()
    result = runner.invoke(cli, ["list-scenarios", "--tier", "easy"])
    assert result.exit_code == 0
    assert "easy" in result.output


def test_export_jsonl(tmp_path):
    runner = CliRunner()
    out = tmp_path / "out.jsonl"
    result = runner.invoke(cli, ["export", "--format", "jsonl", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    sample = json.loads(lines[0])
    assert "id" in sample and "category" in sample


def test_export_csv(tmp_path):
    runner = CliRunner()
    out = tmp_path / "out.csv"
    result = runner.invoke(cli, ["export", "--format", "csv", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "id,category,name" in text


def test_export_huggingface(tmp_path):
    runner = CliRunner()
    out = tmp_path / "hf"
    result = runner.invoke(cli, ["export", "--format", "huggingface", "--output", str(out)])
    assert result.exit_code == 0
    assert (out / "data.jsonl").exists()
    assert (out / "dataset_info.json").exists()
