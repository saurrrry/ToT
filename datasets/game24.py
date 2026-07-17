from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Game24Sample:
    """A single Game of 24 dataset sample."""

    # 数据集中的唯一标识符
    id: str

    # 四个输入数字
    numbers: list[int]

    # 数据集中给出的参考解
    solutions: list[str]

    # 该题是否存在解
    solvable: bool

    # 保存原始数据，方便以后分析 amt、solved_rate 等字段
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
            # 忽略空行。
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
        # 不调用全局 random.seed()。
        #
        # 单独创建一个 Random 对象，可以避免影响程序其他模块中的随机数。
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

    numbers = item["numbers"]

    if not isinstance(numbers, list):
        raise ValueError(
            f"'numbers' must be a list at line {line_number}"
        )

    if len(numbers) != 4:
        raise ValueError(
            f"Expected exactly four numbers at line {line_number}, "
            f"got {len(numbers)}"
        )

    if not all(type(number) is int for number in numbers):
        raise ValueError(
            f"All Game24 numbers must be integers at line {line_number}"
        )

    solutions = item["solutions"]

    if not isinstance(solutions, list):
        raise ValueError(
            f"'solutions' must be a list at line {line_number}"
        )

    # 除去主要字段后，其他字段都作为 metadata 保存。
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
        numbers=numbers,
        solutions=[str(solution) for solution in solutions],
        solvable=bool(item["solvable"]),
        metadata=metadata,
    )