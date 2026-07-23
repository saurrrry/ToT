from __future__ import annotations

import json

from datasets.gsm8k import GSM8KSample
from models.base_model import BaseModel
from prompts.gsm8k_prompt import build_gsm8k_value_prompt
from solvers.base_solver import BaseSolver, SolverResult
from solvers.gsm8k_tot.bfs import bfs_search
from solvers.gsm8k_tot.generator import GSM8KStepGenerator
from solvers.gsm8k_tot.state import GSM8KState
from utils.parsing import parse_state_scores


class GSM8KStateValueEvaluator:
    """Score GSM8K partial reasoning states with the model."""

    def __init__(
        self,
        model: BaseModel,
        *,
        temperature: float = 0.0,
        max_tokens: int = 128,
        batch_size: int = 12,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.seed = seed

        self.cache: dict[tuple[str, ...], float] = {}
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.duration_seconds = 0.0
        self.logs: list[dict] = []

    def score_states(
        self,
        states: list[GSM8KState],
    ) -> list[float]:
        unique_uncached = []
        pending = set()

        for state in states:
            if state.key in self.cache:
                continue

            if state.key in pending:
                continue

            pending.add(state.key)
            unique_uncached.append(state)

        for start in range(
            0,
            len(unique_uncached),
            self.batch_size,
        ):
            batch = unique_uncached[
                start : start + self.batch_size
            ]
            self._evaluate_batch(batch)

        return [
            self.cache.get(state.key, 0.35)
            for state in states
        ]

    def _evaluate_batch(
        self,
        states: list[GSM8KState],
    ) -> None:
        if not states:
            return

        prompt = build_gsm8k_value_prompt(states)

        generation = self.model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed + self.model_calls,
        )

        self.model_calls += 1

        if generation.prompt_tokens is not None:
            self.prompt_tokens += generation.prompt_tokens

        if generation.completion_tokens is not None:
            self.completion_tokens += (
                generation.completion_tokens
            )

        if generation.duration_seconds is not None:
            self.duration_seconds += (
                generation.duration_seconds
            )

        parse_result = parse_state_scores(
            generation.text,
            expected_count=len(states),
        )

        for state, score in zip(
            states,
            parse_result.scores,
            strict=True,
        ):
            self.cache[state.key] = score

        self.logs.append(
            {
                "prompt": prompt,
                "raw_response": generation.text,
                "scores": parse_result.scores,
                "parse_success": parse_result.success,
                "parse_reason": parse_result.reason,
                "ratings": [
                    {
                        "state": state.to_dict(),
                        "score": score,
                    }
                    for state, score in zip(
                        states,
                        parse_result.scores,
                        strict=True,
                    )
                ],
            }
        )


class GSM8KToTBFSSolver(BaseSolver):
    """Beam-BFS Tree-of-Thoughts solver for GSM8K."""

    name = "tot_bfs"

    def __init__(
        self,
        model: BaseModel,
        *,
        generation_temperature: float = 0.7,
        value_temperature: float = 0.0,
        step_max_tokens: int = 512,
        value_max_tokens: int = 128,
        branch_factor: int = 3,
        beam_width: int = 3,
        max_depth: int = 6,
        value_batch_size: int = 12,
        max_expanded_nodes: int = 100,
    ) -> None:
        self.model = model
        self.generation_temperature = generation_temperature
        self.value_temperature = value_temperature
        self.step_max_tokens = step_max_tokens
        self.value_max_tokens = value_max_tokens
        self.branch_factor = branch_factor
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.value_batch_size = value_batch_size
        self.max_expanded_nodes = max_expanded_nodes

    def solve(
        self,
        sample: GSM8KSample,
        *,
        seed: int | None = None,
    ) -> SolverResult:
        selected_seed = 42 if seed is None else seed

        initial_state = GSM8KState.initial(
            sample.question
        )

        generator = GSM8KStepGenerator(
            self.model,
            branch_factor=self.branch_factor,
            temperature=self.generation_temperature,
            max_tokens=self.step_max_tokens,
            seed=selected_seed,
        )

        value_evaluator = GSM8KStateValueEvaluator(
            self.model,
            temperature=self.value_temperature,
            max_tokens=self.value_max_tokens,
            batch_size=self.value_batch_size,
            seed=selected_seed + 10_000,
        )

        search_result = bfs_search(
            initial_state,
            generator.generate_successors,
            value_evaluator.score_states,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
            max_expanded_nodes=self.max_expanded_nodes,
        )

        answer = None
        if search_result.solution is not None:
            answer = search_result.solution.final_answer

        prompt_tokens = (
            generator.prompt_tokens
            + value_evaluator.prompt_tokens
        )
        completion_tokens = (
            generator.completion_tokens
            + value_evaluator.completion_tokens
        )
        duration_seconds = (
            generator.duration_seconds
            + value_evaluator.duration_seconds
        )
        model_calls = (
            generator.model_calls
            + value_evaluator.model_calls
        )

        metadata = {
            "method": self.name,
            "strategy": "bfs",
            "search": {
                "solved": search_result.solved,
                "expanded_nodes": (
                    search_result.expanded_nodes
                ),
                "generated_nodes": (
                    search_result.generated_nodes
                ),
                "trace": search_result.trace,
                "solution_steps": (
                    list(search_result.solution.steps)
                    if search_result.solution is not None
                    else []
                ),
            },
            "generator": {
                "model_calls": generator.model_calls,
                "logs": generator.logs,
            },
            "value_evaluator": {
                "model_calls": (
                    value_evaluator.model_calls
                ),
                "cached_states": len(
                    value_evaluator.cache
                ),
                "logs": value_evaluator.logs,
            },
        }

        raw_response = json.dumps(
            {
                "generator_logs": generator.logs,
                "value_logs": value_evaluator.logs,
            },
            ensure_ascii=False,
        )

        return SolverResult(
            expression=answer,
            raw_response=raw_response,
            prompt="",
            duration_seconds=duration_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_calls=model_calls,
            metadata=metadata,
        )
