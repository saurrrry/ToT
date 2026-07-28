"""Shared model backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    """Standard result returned by every model backend."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] | None = None


class BaseModel(ABC):
    """Abstract interface for all language-model backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        context_length: int | None = None,
    ) -> GenerationResult:
        """
        Generate one response for a prompt.

        Every model backend must implement this method.
        """
        raise NotImplementedError
