import pytest

from tom_benchmark.adapters import (
    MODEL_REGISTRY,
    ClaudeAdapter,
    OpenAIAdapter,
    get_adapter,
)


def test_registry_routes_claude_models():
    adapter = get_adapter("claude-sonnet-4-6")
    assert isinstance(adapter, ClaudeAdapter)


def test_registry_routes_openai_models():
    adapter = get_adapter("gpt-4o")
    assert isinstance(adapter, OpenAIAdapter)


def test_unknown_model_falls_through_prefix_for_claude_named():
    adapter = get_adapter("claude-something-new")
    assert isinstance(adapter, ClaudeAdapter)


def test_truly_unknown_model_raises():
    with pytest.raises(ValueError):
        get_adapter("totallyfakemodel-xyz")


def test_claude_adapter_is_available_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = ClaudeAdapter("claude-sonnet-4-6")
    assert adapter.is_available() is False


def test_claude_adapter_is_available_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    adapter = ClaudeAdapter("claude-sonnet-4-6")
    assert adapter.is_available() is True


def test_openai_adapter_is_available_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    adapter = OpenAIAdapter("gpt-4o")
    assert adapter.is_available() is True


def test_registry_contents_match_expected():
    # Sanity check: all registered models map to a known provider.
    for model, provider in MODEL_REGISTRY.items():
        assert provider in {"anthropic", "openai"}
