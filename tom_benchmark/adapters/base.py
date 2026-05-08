"""Abstract base class and shared types for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    text: str
    model: str
    latency_seconds: float
    raw: object | None = None


class LLMAdapter(ABC):
    """Strategy interface for LLM providers."""

    model: str

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def query(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Send a prompt and return a normalized response."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this adapter has the credentials it needs to make a call."""
