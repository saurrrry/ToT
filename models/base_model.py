from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    """Standard result returned by every model backend."""

    # 模型最终返回的文本。
    text: str

    # 实际使用的模型名称。
    model: str

    # 输入 token 数量。
    # 某些模型后端可能不返回，因此允许为 None。
    prompt_tokens: int | None = None

    # 输出 token 数量。
    completion_tokens: int | None = None

    # 模型调用耗时，单位为秒。
    duration_seconds: float | None = None

    # 保存模型后端返回的其他信息。
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
    ) -> GenerationResult:
        """
        Generate one response for a prompt.

        Every model backend must implement this method.
        """
        raise NotImplementedError