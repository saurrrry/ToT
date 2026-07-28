"""Exact verifier for Game24 arithmetic expressions."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable


ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    value: Fraction | None = None
    reason: str = ""


def verify_24_expression(
    expression: str,
    numbers: Iterable[int],
    target: int = 24,
) -> VerificationResult:
    """Verify a standard Game of 24 expression."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return VerificationResult(
            False,
            reason=f"syntax error: {exc.msg}",
        )

    expected_numbers = Counter(
        Fraction(number, 1) for number in numbers
    )
    used_numbers: list[Fraction] = []

    try:
        value = _eval_node(tree.body, used_numbers)
    except ZeroDivisionError:
        return VerificationResult(
            False,
            reason="division by zero",
        )
    except ValueError as exc:
        return VerificationResult(
            False,
            reason=str(exc),
        )

    actual_numbers = Counter(used_numbers)

    if actual_numbers != expected_numbers:
        return VerificationResult(
            False,
            value=value,
            reason=(
                f"number mismatch: expected {expected_numbers}, "
                f"got {actual_numbers}"
            ),
        )

    if value != Fraction(target, 1):
        return VerificationResult(
            False,
            value=value,
            reason=f"value is {value}, not {target}",
        )

    return VerificationResult(
        True,
        value=value,
        reason="valid",
    )


def _eval_node(
    node: ast.AST,
    used_numbers: list[Fraction],
) -> Fraction:
    # Only integer literals are valid input numbers.
    if isinstance(node, ast.Constant) and type(node.value) is int:
        value = Fraction(node.value, 1)
        used_numbers.append(value)
        return value

    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        ALLOWED_BINOPS,
    ):
        left = _eval_node(node.left, used_numbers)
        right = _eval_node(node.right, used_numbers)

        if isinstance(node.op, ast.Add):
            return left + right

        if isinstance(node.op, ast.Sub):
            return left - right

        if isinstance(node.op, ast.Mult):
            return left * right

        if right == 0:
            raise ZeroDivisionError

        return left / right

    raise ValueError(
        f"unsupported expression element: "
        f"{type(node).__name__}"
    )
