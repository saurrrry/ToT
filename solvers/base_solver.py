"""Common solver interfaces and result schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SolverResult:
    """Standard output produced by every solving method."""

    expression: str | None

    raw_response: str

    prompt: str

    duration_seconds: float | None

    prompt_tokens: int | None

    completion_tokens: int | None

    model_calls: int

    metadata: dict[str, Any]


class BaseSolver(ABC):
    """Common interface for baseline, CoT, ToT and MCTS."""

    name: str

    @abstractmethod
    def solve(
        self,
        sample: Any,
        *,
        seed: int | None = None,
    ) -> SolverResult:
        """Solve one dataset sample."""
        raise NotImplementedError
