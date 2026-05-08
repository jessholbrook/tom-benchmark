"""Adapter factory and model registry."""

from __future__ import annotations

from .base import LLMAdapter, LLMResponse
from .claude import ClaudeAdapter
from .openai import OpenAIAdapter

# Registry of supported model names → provider key.
MODEL_REGISTRY: dict[str, str] = {
    # Anthropic
    "claude-sonnet-4-6": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
    # OpenAI
    "gpt-4o": "openai",
    "gpt-4-turbo": "openai",
    "gpt-3.5-turbo": "openai",
}

PROVIDER_PREFIXES: list[tuple[str, str]] = [
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
]


def _resolve_provider(model_name: str) -> str:
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    for prefix, provider in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            return provider
    raise ValueError(
        f"Unknown model: {model_name!r}. Add it to MODEL_REGISTRY in tom_benchmark/adapters/__init__.py."
    )


def get_adapter(model_name: str) -> LLMAdapter:
    """Return an adapter instance for the given model name."""
    provider = _resolve_provider(model_name)
    if provider == "anthropic":
        return ClaudeAdapter(model_name)
    if provider == "openai":
        return OpenAIAdapter(model_name)
    raise ValueError(f"Unsupported provider: {provider!r}")


__all__ = [
    "LLMAdapter",
    "LLMResponse",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "MODEL_REGISTRY",
    "get_adapter",
]
