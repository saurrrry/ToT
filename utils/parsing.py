"""Parsing helpers for model answers and state scores."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


# Match the final answer line without swallowing earlier CoT steps.
ANSWER_PATTERN = re.compile(
    r"Answer\s*:\s*"
    r"(?P<expression>[^\r\n]+?)"
    r"\s*=\s*24\b",
    flags=re.IGNORECASE,
)


# Fallback for models that only emit a final expression line.
FINAL_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<expression>[0-9+\-*/().\s]+?)"
    r"\s*=\s*24\b",
    flags=(
        re.IGNORECASE
        | re.MULTILINE
    ),
)


def extract_final_expression(
    response: str,
) -> str | None:
    """Extract the final Game24 expression from baseline or CoT output."""
    normalized = _normalize_symbols(response)

    # CoT can contain many equations; the final answer should be last.
    answer_matches = list(
        ANSWER_PATTERN.finditer(normalized)
    )

    if answer_matches:
        expression = (
            answer_matches[-1]
            .group("expression")
        )

        expression = _clean_expression(
            expression
        )

        if _looks_like_pure_expression(
            expression
        ):
            return expression

        return None

    final_line_matches = list(
        FINAL_LINE_PATTERN.finditer(
            normalized
        )
    )

    if final_line_matches:
        expression = (
            final_line_matches[-1]
            .group("expression")
        )

        expression = _clean_expression(
            expression
        )

        if _looks_like_pure_expression(
            expression
        ):
            return expression

        return None

    # Accept bare expressions only when the entire output is arithmetic.
    stripped = normalized.strip()

    if _looks_like_pure_expression(
        stripped
    ):
        return _clean_expression(
            stripped
        )

    return None


def _normalize_symbols(
    text: str,
) -> str:
    """Normalize Unicode math symbols to verifier-supported operators."""
    return (
        text
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("＋", "+")
        .replace("／", "/")
        .replace("＊", "*")
    )


def _clean_expression(
    expression: str,
) -> str:
    """Remove answer labels, trailing target text, and extra whitespace."""
    expression = expression.strip()

    # Some models repeat the prompt suffix before the actual expression.
    expression = re.sub(
        r"^\s*Answer\s*:\s*",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    expression = re.sub(
        r"\s*=\s*24\s*$",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    return expression.strip()


def _looks_like_pure_expression(
    text: str,
) -> bool:
    """Check whether text is shaped like a single arithmetic expression."""
    if not text:
        return False

    # Multi-line text is likely reasoning, not a final expression.
    if "\n" in text or "\r" in text:
        return False

    # This is only a syntax filter; semantic checks stay in the verifier.
    return (
        re.fullmatch(
            r"[0-9+\-*/().\s]+",
            text,
        )
        is not None
    )


VALID_STATE_RATINGS = {
    "sure",
    "likely",
    "maybe",
    "impossible",
}

RATING_TO_SCORE = {
    "sure": 1.0,
    "likely": 0.75,
    "maybe": 0.35,
    "impossible": 0.0,
}


@dataclass(frozen=True)
class ScoreParseResult:
    scores: list[float]
    success: bool
    reason: str


def parse_state_scores(
    response: str,
    expected_count: int,
) -> ScoreParseResult:
    """
    Parse compact ToT state scores.

    Preferred model output:

        {"scores":[1.0,0.75,0.35,0.0]}

    Older rating output is still accepted as a fallback.
    """
    if expected_count < 0:
        raise ValueError(
            "expected_count must be non-negative"
        )

    if expected_count == 0:
        return ScoreParseResult(
            scores=[],
            success=True,
            reason="empty batch",
        )

    cleaned = _remove_code_fence(
        response.strip()
    )

    data = _parse_json_object(cleaned)

    if data is None:
        return ScoreParseResult(
            scores=_default_scores(expected_count),
            success=False,
            reason="invalid JSON object",
        )

    scores = _parse_scores_array(
        data.get("scores"),
        expected_count,
    )

    if scores is not None:
        return ScoreParseResult(
            scores=scores,
            success=True,
            reason="valid scores",
        )

    ratings = _parse_state_ratings_result(
        response,
        expected_count,
    )

    if ratings.success:
        return ScoreParseResult(
            scores=[
                RATING_TO_SCORE[rating]
                for rating in ratings.ratings
            ],
            success=True,
            reason="valid legacy ratings",
        )

    return ScoreParseResult(
        scores=_default_scores(expected_count),
        success=False,
        reason=(
            "scores must contain exactly "
            f"{expected_count} numeric values in [0, 1]"
        ),
    )


def _parse_scores_array(
    raw_scores: Any,
    expected_count: int,
) -> list[float] | None:
    if not isinstance(raw_scores, list):
        return None

    if len(raw_scores) != expected_count:
        return None

    scores: list[float] = []

    for raw_score in raw_scores:
        if isinstance(raw_score, bool):
            return None

        if not isinstance(raw_score, (int, float)):
            return None

        score = float(raw_score)

        if not 0.0 <= score <= 1.0:
            return None

        scores.append(score)

    return scores


@dataclass(frozen=True)
class RatingParseResult:
    ratings: list[str]
    success: bool
    reason: str


def parse_state_ratings(
    response: str,
    expected_count: int,
) -> list[str]:
    """Parse legacy state ratings, defaulting invalid items to "maybe"."""
    return _parse_state_ratings_result(
        response,
        expected_count,
    ).ratings


def _parse_state_ratings_result(
    response: str,
    expected_count: int,
) -> RatingParseResult:
    if expected_count < 0:
        raise ValueError(
            "expected_count must be non-negative"
        )

    if expected_count == 0:
        return RatingParseResult(
            ratings=[],
            success=True,
            reason="empty batch",
        )

    cleaned = _remove_code_fence(
        response.strip()
    )

    data = _parse_json_object(cleaned)

    if data is None:
        return RatingParseResult(
            ratings=_default_ratings(expected_count),
            success=False,
            reason="invalid JSON object",
        )

    raw_ratings = data.get("ratings")

    if not isinstance(raw_ratings, list):
        return RatingParseResult(
            ratings=_default_ratings(expected_count),
            success=False,
            reason="missing ratings list",
        )

    if len(raw_ratings) != expected_count:
        return RatingParseResult(
            ratings=_default_ratings(expected_count),
            success=False,
            reason=(
                "ratings must contain exactly "
                f"{expected_count} items"
            ),
        )

    parsed_by_id: dict[int, str] = {}

    for item in raw_ratings:
        if not isinstance(item, dict):
            continue

        state_id = item.get("id")
        rating = item.get("rating")

        # bool is an int subclass, so exclude it explicitly.
        if (
            not isinstance(state_id, int)
            or isinstance(state_id, bool)
        ):
            continue

        if not isinstance(rating, str):
            continue

        normalized_rating = (
            rating
            .strip()
            .lower()
        )

        if (
            normalized_rating
            not in VALID_STATE_RATINGS
        ):
            continue

        # Ignore IDs outside the current batch.
        if not (
            1
            <= state_id
            <= expected_count
        ):
            continue

        # Later duplicates overwrite earlier ones.
        parsed_by_id[state_id] = (
            normalized_rating
        )

    ratings = [
        parsed_by_id.get(
            state_id,
            "maybe",
        )
        for state_id in range(
            1,
            expected_count + 1,
        )
    ]

    success = len(parsed_by_id) == expected_count

    return RatingParseResult(
        ratings=ratings,
        success=success,
        reason="valid ratings" if success else "missing or invalid rating",
    )


def _remove_code_fence(
    text: str,
) -> str:
    """Strip a surrounding Markdown JSON fence if the model adds one."""
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    return text.strip()


def _parse_json_object(
    text: str,
) -> dict[str, Any] | None:
    """Parse a JSON object, allowing small text before or after it."""
    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        first_brace = text.find("{")
        last_brace = text.rfind("}")

        if (
            first_brace == -1
            or last_brace == -1
            or first_brace >= last_brace
        ):
            return None

        json_fragment = text[
            first_brace:
            last_brace + 1
        ]

        try:
            data = json.loads(
                json_fragment
            )
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    return data


def _default_ratings(
    count: int,
) -> list[str]:
    """Use neutral ratings when the model output cannot be trusted."""
    return [
        "maybe"
        for _ in range(count)
    ]


def _default_scores(
    count: int,
) -> list[float]:
    return [
        RATING_TO_SCORE["maybe"]
        for _ in range(count)
    ]
