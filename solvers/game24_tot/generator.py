"""Programmatic successor generation for Game24 states."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from solvers.game24_tot.state import (
    State,
    Term,
    create_state,
    format_fraction,
)


def generate_successors(
    state: State,
) -> list[State]:
    """Generate every legal successor state without model involvement."""
    if len(state.terms) < 2:
        return []

    successors: list[State] = []

    # Deduplicate equivalent children from the same parent state.
    seen_keys: set[
        tuple[tuple[int, int], ...]
    ] = set()

    for left_index, right_index in combinations(
        range(len(state.terms)),
        2,
    ):
        left = state.terms[left_index]
        right = state.terms[right_index]

        remaining_terms = [
            term
            for index, term in enumerate(state.terms)
            if index not in {
                left_index,
                right_index,
            }
        ]

        operations = _generate_operations(
            left,
            right,
        )

        for result_value, result_expression, step in operations:
            new_term = Term(
                value=result_value,
                expression=result_expression,
            )

            child = create_state(
                terms=[
                    *remaining_terms,
                    new_term,
                ],
                steps=(
                    *state.steps,
                    step,
                ),
            )

            if child.key in seen_keys:
                continue

            seen_keys.add(child.key)
            successors.append(child)

    return successors


def _generate_operations(
    left: Term,
    right: Term,
) -> list[
    tuple[
        Fraction,
        str,
        str,
    ]
]:
    """Generate all legal operations between two terms."""
    operations: list[
        tuple[
            Fraction,
            str,
            str,
        ]
    ] = []

    left_value = left.value
    right_value = right.value

    left_expression = left.expression
    right_expression = right.expression

    # Addition and multiplication are commutative, so one order is enough.
    add_value = left_value + right_value

    operations.append(
        (
            add_value,
            (
                f"({left_expression} "
                f"+ {right_expression})"
            ),
            _format_step(
                left,
                "+",
                right,
                add_value,
            ),
        )
    )

    multiply_value = left_value * right_value

    operations.append(
        (
            multiply_value,
            (
                f"({left_expression} "
                f"* {right_expression})"
            ),
            _format_step(
                left,
                "*",
                right,
                multiply_value,
            ),
        )
    )

    # Subtraction and division need both operand orders.
    subtract_lr = left_value - right_value

    operations.append(
        (
            subtract_lr,
            (
                f"({left_expression} "
                f"- {right_expression})"
            ),
            _format_step(
                left,
                "-",
                right,
                subtract_lr,
            ),
        )
    )

    subtract_rl = right_value - left_value

    operations.append(
        (
            subtract_rl,
            (
                f"({right_expression} "
                f"- {left_expression})"
            ),
            _format_step(
                right,
                "-",
                left,
                subtract_rl,
            ),
        )
    )

    if right_value != 0:
        divide_lr = left_value / right_value

        operations.append(
            (
                divide_lr,
                (
                    f"({left_expression} "
                    f"/ {right_expression})"
                ),
                _format_step(
                    left,
                    "/",
                    right,
                    divide_lr,
                ),
            )
        )

    if left_value != 0:
        divide_rl = right_value / left_value

        operations.append(
            (
                divide_rl,
                (
                    f"({right_expression} "
                    f"/ {left_expression})"
                ),
                _format_step(
                    right,
                    "/",
                    left,
                    divide_rl,
                ),
            )
        )

    return operations


def _format_step(
    left: Term,
    operator: str,
    right: Term,
    result: Fraction,
) -> str:
    """Format one arithmetic step for logs and result JSON."""
    return (
        f"{format_fraction(left.value)} "
        f"{operator} "
        f"{format_fraction(right.value)} "
        f"= {format_fraction(result)}"
    )
