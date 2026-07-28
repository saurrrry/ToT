"""Chain-of-thought baseline solvers."""

from __future__ import annotations

import json
from collections import Counter

from datasets.game24 import Game24Sample
from datasets.gsm8k import GSM8KSample
from models.base_model import BaseModel
from prompts.game24_prompt import build_cot_prompt
from prompts.gsm8k_prompt import build_gsm8k_cot_prompt
from solvers.base_solver import BaseSolver, SolverResult
from utils.parsing import extract_final_expression
from verifier.gsm8k import (
    extract_gsm8k_final_answer,
    normalize_gsm8k_answer,
)


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


class GSM8KCoTSolver(BaseSolver):
    """Few-shot chain-of-thought solver for GSM8K."""

    name = "cot"

    def __init__(
        self,
        model: BaseModel,
        *,
        temperature: float = 0.0,
        max_tokens: int = 768,
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
        prompt = build_gsm8k_cot_prompt(
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


class GSM8KSelfConsistencyCoTSolver(BaseSolver):
    """Sample multiple CoT answers and return the majority vote."""

    name = "self_consistency_cot"

    def __init__(
        self,
        model: BaseModel,
        *,
        temperature: float = 0.7,
        max_tokens: int = 768,
        samples: int = 5,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.samples = samples

    def solve(
        self,
        sample: GSM8KSample,
        *,
        seed: int | None = None,
    ) -> SolverResult:
        prompt = build_gsm8k_cot_prompt(
            sample.question
        )

        generations = []
        answers = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_duration_seconds = 0.0

        for index in range(self.samples):
            generation_seed = (
                None
                if seed is None
                else seed + index
            )

            generation = self.model.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=generation_seed,
            )

            answer = extract_gsm8k_final_answer(
                generation.text
            )

            normalized_answer = (
                normalize_gsm8k_answer(answer)
                if answer is not None
                else None
            )

            generations.append(
                {
                    "index": index + 1,
                    "seed": generation_seed,
                    "raw_response": generation.text,
                    "answer": answer,
                    "normalized_answer": normalized_answer,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": (
                        generation.completion_tokens
                    ),
                    "duration_seconds": (
                        generation.duration_seconds
                    ),
                }
            )

            if normalized_answer is not None:
                answers.append(normalized_answer)

            if generation.prompt_tokens is not None:
                total_prompt_tokens += (
                    generation.prompt_tokens
                )

            if generation.completion_tokens is not None:
                total_completion_tokens += (
                    generation.completion_tokens
                )

            if generation.duration_seconds is not None:
                total_duration_seconds += (
                    generation.duration_seconds
                )

        selected_answer = _majority_vote(answers)

        return SolverResult(
            expression=selected_answer,
            raw_response=json.dumps(
                generations,
                ensure_ascii=False,
            ),
            prompt=prompt,
            duration_seconds=total_duration_seconds,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            model_calls=self.samples,
            metadata={
                "method": self.name,
                "samples": self.samples,
                "temperature": self.temperature,
                "generations": generations,
                "votes": dict(Counter(answers)),
            },
        )


def _majority_vote(
    answers: list[str],
) -> str | None:
    if not answers:
        return None

    counts = Counter(answers)

    return counts.most_common(1)[0][0]
