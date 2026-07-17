from __future__ import annotations

from ..datasets.game24 import Game24Sample
from ..models.base_model import BaseModel
from ..prompts.game24_prompt import build_cot_prompt
from ..utils.parsing import extract_final_expression
from .base_solver import BaseSolver, SolverResult


class CoTSolver(BaseSolver):
    """Five-shot chain-of-thought baseline."""

    name = "cot"

    def __init__(
        self,
        model: BaseModel,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
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
        prompt = build_cot_prompt(
            sample.input_text
        )

        generation = self.model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )

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
