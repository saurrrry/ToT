from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GSM8KSample:
    """A single GSM8K sample."""

    id: str
    question: str
    answer: str
    final_answer: str
    metadata: dict[str, Any]

    @property
    def input_text(self) -> str:
        return self.question


def load_gsm8k_dataset(
    path: str | Path,
    *,
    shuffle: bool = True,
    seed: int = 42,
    limit: int | None = None,
) -> list[GSM8KSample]:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"GSM8K dataset was not found: {dataset_path}"
        )

    items = _load_items(dataset_path)
    samples = [
        _parse_sample(item, index)
        for index, item in enumerate(items, start=1)
    ]

    # The first five records are used as fixed few-shot
    # examples in prompts/gsm8k_prompt.py. Exclude them from
    # evaluation before shuffling and applying --limit.
    samples = samples[5:]

    if shuffle:
        random_generator = random.Random(seed)
        random_generator.shuffle(samples)

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")

        samples = samples[:limit]

    return samples


def _load_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                "GSM8K JSON file must contain a list of objects"
            )

        return data

    items: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected JSON object at line {line_number}"
            )

        items.append(item)

    return items


def _parse_sample(
    item: dict[str, Any],
    index: int,
) -> GSM8KSample:
    if "question" not in item:
        raise ValueError(
            f"Missing question in GSM8K item {index}"
        )

    if "answer" not in item:
        raise ValueError(
            f"Missing answer in GSM8K item {index}"
        )

    question = str(item["question"]).strip()
    answer = str(item["answer"]).strip()

    final_answer = str(
        item.get(
            "final_answer",
            _extract_final_answer(answer),
        )
    ).strip()

    metadata = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "id",
            "question",
            "answer",
            "final_answer",
        }
    }

    return GSM8KSample(
        id=str(item.get("id", f"gsm8k-{index:04d}")),
        question=question,
        answer=answer,
        final_answer=final_answer,
        metadata=metadata,
    )


def _extract_final_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        return answer.strip()

    return answer.rsplit(marker, 1)[1].strip()
