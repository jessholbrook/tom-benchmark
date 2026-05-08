"""OpenAI GPT adapter."""

from __future__ import annotations

import os
import time

from .base import LLMAdapter, LLMResponse


class OpenAIAdapter(LLMAdapter):
    """Wraps the OpenAI Python SDK with lazy client initialization."""

    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        super().__init__(model)
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "openai package is not installed. Install with `pip install openai`."
                ) from e
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to your environment or .env file."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def query(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        latency = time.perf_counter() - start

        text = ""
        if resp.choices:
            text = (resp.choices[0].message.content or "").strip()

        return LLMResponse(text=text, model=self.model, latency_seconds=latency, raw=resp)
