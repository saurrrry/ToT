from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..datasets.game24 import Game24Sample


@dataclass(frozen=True)
class SolverResult:
    """Standard output produced by every solving method."""

    # 从模型输出中提取出的最终表达式。
    expression: str | None

    # 模型的完整原始输出。
    raw_response: str

    # 实际发送给模型的 prompt。
    prompt: str

    # 模型调用的耗时。
    duration_seconds: float | None

    # 输入 token 数。
    prompt_tokens: int | None

    # 输出 token 数。
    completion_tokens: int | None

    # 本次 solve 实际调用模型的次数。
    model_calls: int

    # 以后 ToT/MCTS 可以在这里保存搜索树信息。
    metadata: dict[str, Any]


class BaseSolver(ABC):
    """Common interface for baseline, CoT, ToT and MCTS."""

    name: str

    @abstractmethod
    def solve(
        self,
        sample: Game24Sample,
        *,
        seed: int | None = None,
    ) -> SolverResult:
        """Solve one Game of 24 sample."""
        raise NotImplementedError
