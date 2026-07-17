from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from .state import (
    State,
    Term,
    create_state,
    format_fraction,
)


def generate_successors(
    state: State,
) -> list[State]:
    """
    程序化生成当前状态的所有合法子状态。

    每次：

    1. 选择两个剩余项；
    2. 对它们执行一种四则运算；
    3. 删除原来的两个项；
    4. 加入新产生的项。

    模型不参与候选操作生成。
    """
    if len(state.terms) < 2:
        return []

    successors: list[State] = []

    # 在同一个父状态内，根据剩余值去重。
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
    """
    生成两个项之间所有合法运算。

    返回值中每个元素包含：

        result_value
        result_expression
        readable_step
    """
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

    # 加法满足交换律，因此只生成一个方向。
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

    # 乘法满足交换律，因此只生成一个方向。
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

    # 减法不满足交换律，需要生成两个方向。
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

    # 除法需要避免除以零。
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
    """
    生成方便记录和输出的步骤文本。
    """
    return (
        f"{format_fraction(left.value)} "
        f"{operator} "
        f"{format_fraction(right.value)} "
        f"= {format_fraction(result)}"
    )
