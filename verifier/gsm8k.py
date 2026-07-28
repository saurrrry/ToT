"""Answer extraction and verification for GSM8K."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


FINAL_ANSWER_PATTERN = re.compile(
    r"Answer\s*:\s*(?P<answer>[^\r\n]+)",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?"
)


@dataclass(frozen=True)
class GSM8KVerificationResult:
    ok: bool
    predicted: str | None
    expected: str
    reason: str


def extract_gsm8k_final_answer(
    text: str,
) -> str | None:
    if "####" in text:
        return None

    matches = list(
        FINAL_ANSWER_PATTERN.finditer(text)
    )

    if matches:
        return (
            matches[-1]
            .group("answer")
            .strip()
        )

    number_matches = list(
        NUMBER_PATTERN.finditer(text)
    )

    if not number_matches:
        return None

    return number_matches[-1].group(0).strip()


def normalize_gsm8k_answer(
    answer: str,
) -> str:
    cleaned = answer.strip()

    if cleaned.startswith("####"):
        return cleaned.lower()

    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned.split(":", 1)[1].strip()

    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.strip()

    match = NUMBER_PATTERN.search(cleaned)
    if match is not None:
        cleaned = match.group(0)
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("$", "")

    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]

    decimal_value = _to_decimal(cleaned)

    if decimal_value is None:
        return cleaned.lower()

    normalized = decimal_value.normalize()

    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))

    return format(normalized, "f").rstrip("0").rstrip(".")


def verify_gsm8k_answer(
    prediction: str | None,
    expected: str,
) -> GSM8KVerificationResult:
    expected_normalized = normalize_gsm8k_answer(
        expected
    )

    if prediction is None:
        return GSM8KVerificationResult(
            ok=False,
            predicted=None,
            expected=expected_normalized,
            reason="could not extract final answer",
        )

    predicted_normalized = normalize_gsm8k_answer(
        prediction
    )

    ok = predicted_normalized == expected_normalized

    return GSM8KVerificationResult(
        ok=ok,
        predicted=predicted_normalized,
        expected=expected_normalized,
        reason="valid" if ok else "answer mismatch",
    )


def _to_decimal(
    text: str,
) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None
