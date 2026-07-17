from __future__ import annotations

from ..datasets.game24 import Game24Sample
from ..models.base_model import BaseModel
from ..prompts.game24_prompt import build_baseline_prompt
from ..utils.parsing import extract_final_expression
from .base_solver import BaseSolver, SolverResult


class BaselineSolver(BaseSolver):
    """Five-shot direct-answer baseline."""

    name = "baseline"

    def __init__(
        self,
        model: BaseModel,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(
        self,
        sample: Game24Sample,
        *,
        seed: int | None = None,
    ) -> SolverResult:
        # 根据当前题目的四个数字创建 prompt。
        prompt = build_baseline_prompt(
            sample.input_text
        )

        # 调用模型。
        generation = self.model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )

        # 从模型返回文本中提取最终算术表达式。
        expression = extract_final_expression(
            generation.text
        )

        return SolverResult(
            expression=expression,
            raw_response=generation.text,
            prompt=prompt,
            duration_seconds=generation.duration_seconds,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            model_calls=1,
            metadata={
                "method": self.name,
                "model": generation.model,
            },
        )
