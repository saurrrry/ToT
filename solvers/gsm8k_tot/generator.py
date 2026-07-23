from __future__ import annotations

import json
import re

from models.base_model import BaseModel
from prompts.gsm8k_prompt import (
    build_gsm8k_step_generation_prompt,
)
from solvers.gsm8k_tot.state import GSM8KState


class GSM8KStepGenerator:
    """Generate candidate next reasoning steps with the model."""

    def __init__(
        self,
        model: BaseModel,
        *,
        branch_factor: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.branch_factor = branch_factor
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.duration_seconds = 0.0
        self.logs: list[dict] = []

    def generate_successors(
        self,
        state: GSM8KState,
    ) -> list[GSM8KState]:
        if state.is_terminal():
            return []

        prompt = build_gsm8k_step_generation_prompt(
            question=state.question,
            steps=list(state.steps),
            branch_factor=self.branch_factor,
        )

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

        steps = _parse_generated_steps(
            generation.text,
            limit=self.branch_factor,
        )

        children = []
        seen = set()

        for step in steps:
            child = state.add_step(step)

            if child.key in seen:
                continue

            seen.add(child.key)
            children.append(child)

        self.logs.append(
            {
                "prompt": prompt,
                "raw_response": generation.text,
                "steps": steps,
                "children": [
                    child.to_dict()
                    for child in children
                ],
            }
        )

        return children


def _parse_generated_steps(
    text: str,
    *,
    limit: int,
) -> list[str]:
    data = _parse_json_object(text)

    if isinstance(data, dict):
        raw_steps = data.get("steps")

        if isinstance(raw_steps, list):
            steps = [
                str(step).strip()
                for step in raw_steps
                if str(step).strip()
            ]

            return steps[:limit]

    fallback_steps = []

    for line in text.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(
            r"^[-*\d.)\s]+",
            "",
            cleaned,
        ).strip()

        if cleaned:
            fallback_steps.append(cleaned)

    return fallback_steps[:limit]


def _parse_json_object(
    text: str,
) -> dict | None:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    ).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        first = cleaned.find("{")
        last = cleaned.rfind("}")

        if first == -1 or last == -1 or first >= last:
            return None

        try:
            data = json.loads(cleaned[first : last + 1])
        except json.JSONDecodeError:
            return None

    return data if isinstance(data, dict) else None
