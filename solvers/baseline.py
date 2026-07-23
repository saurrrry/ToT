from __future__ import annotations

from datasets.game24 import Game24Sample
from datasets.gsm8k import GSM8KSample
from models.base_model import BaseModel
from prompts.game24_prompt import build_baseline_prompt
from prompts.gsm8k_prompt import build_gsm8k_baseline_prompt
from solvers.base_solver import BaseSolver, SolverResult
from utils.parsing import extract_final_expression
from verifier.gsm8k import extract_gsm8k_final_answer


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


class GSM8KBaselineSolver(BaseSolver):
    """Few-shot direct-answer baseline for GSM8K."""

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
        sample: GSM8KSample,
        *,
        seed: int | None = None,
    ) -> SolverResult:
        prompt = build_gsm8k_baseline_prompt(
            sample.question
        )

        generation = self.model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )

        answer = extract_gsm8k_final_answer(
            generation.text
        )

        return SolverResult(
            expression=answer,
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
