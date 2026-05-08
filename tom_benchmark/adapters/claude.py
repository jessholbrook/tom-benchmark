"""Anthropic Claude adapter."""

from __future__ import annotations

import os
import time

from .base import LLMAdapter, LLMResponse


class ClaudeAdapter(LLMAdapter):
    """Wraps the Anthropic Python SDK with lazy client initialization."""

    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        super().__init__(model)
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic package is not installed. Install with `pip install anthropic`."
                ) from e
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to your environment or .env file."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def query(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        start = time.perf_counter()
        msg = client.messages.create(**kwargs)
        latency = time.perf_counter() - start

        text_parts: list[str] = []
        for block in getattr(msg, "content", []) or []:
            piece = getattr(block, "text", None)
            if piece:
                text_parts.append(piece)
        text = "".join(text_parts).strip()

        return LLMResponse(text=text, model=self.model, latency_seconds=latency, raw=msg)
