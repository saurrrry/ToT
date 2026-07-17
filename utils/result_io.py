from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


def save_json(
    data: Any,
    path: str | Path,
) -> Path:
    """Save experiment data as formatted JSON."""
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )

    return output_path


def _json_default(value: Any) -> Any:
    """Convert otherwise non-serializable values."""
    if isinstance(value, Fraction):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "text": str(value),
        }

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return asdict(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        f"is not JSON serializable"
    )