"""Game24 dataset loading and validation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Game24Sample:
    """A single Game of 24 dataset sample."""

    id: str

    numbers: tuple[int, int, int, int]

    solutions: tuple[str, ...]

    solvable: bool

    metadata: dict[str, Any]

    @property
    def input_text(self) -> str:
        """Convert [4, 4, 6, 8] into '4 4 6 8'."""
        return " ".join(str(number) for number in self.numbers)


def load_game24_dataset(
    path: str | Path,
    *,
    shuffle: bool = True,
    seed: int = 42,
    limit: int | None = None,
    solvable_only: bool = True,
) -> list[Game24Sample]:
    """
    Load Game of 24 samples from a JSONL file.

    Args:
        path:
            JSONL dataset path.

        shuffle:
            Whether to shuffle samples before selecting the limit.

        seed:
            Random seed used for reproducible shuffling.

        limit:
            Maximum number of samples to return.
            None means return all samples.

        solvable_only:
            Whether to keep only samples where solvable is true.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Game24 dataset was not found: {dataset_path}"
        )

    samples: list[Game24Sample] = []

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            # Skip blank lines in JSONL files.
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            sample = _parse_sample(item, line_number)

            if solvable_only and not sample.solvable:
                continue

            samples.append(sample)

    if shuffle:
        # Use a local RNG so loading data does not affect other modules.
        random_generator = random.Random(seed)
        random_generator.shuffle(samples)

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")

        samples = samples[:limit]

    return samples


def _parse_sample(
    item: dict[str, Any],
    line_number: int,
) -> Game24Sample:
    """Validate and convert one JSON object into Game24Sample."""
    required_fields = {
        "id",
        "numbers",
        "solutions",
        "solvable",
    }

    missing_fields = required_fields - item.keys()

    if missing_fields:
        raise ValueError(
            f"Missing fields at line {line_number}: "
            f"{sorted(missing_fields)}"
        )

    raw_numbers = item["numbers"]

    if not isinstance(raw_numbers, list):
        raise ValueError(
            f"'numbers' must be a list at line {line_number}"
        )

    if len(raw_numbers) != 4:
        raise ValueError(
            f"Expected exactly four numbers at line {line_number}, "
            f"got {len(raw_numbers)}"
        )

    if not all(type(number) is int for number in raw_numbers):
        raise ValueError(
            f"All Game24 numbers must be integers at line {line_number}"
        )

    raw_solutions = item["solutions"]

    if not isinstance(raw_solutions, list):
        raise ValueError(
            f"'solutions' must be a list at line {line_number}"
        )

    raw_solvable = item["solvable"]

    if type(raw_solvable) is not bool:
        raise ValueError(
            f"'solvable' must be bool at line {line_number}, "
            f"got {raw_solvable!r}"
        )

    # Preserve dataset-specific fields for later analysis.
    metadata = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "id",
            "numbers",
            "solutions",
            "solvable",
        }
    }

    return Game24Sample(
        id=str(item["id"]),
        numbers=(
            raw_numbers[0],
            raw_numbers[1],
            raw_numbers[2],
            raw_numbers[3],
        ),
        solutions=tuple(
            str(solution)
            for solution in raw_solutions
        ),
        solvable=raw_solvable,
        metadata=metadata,
    )
