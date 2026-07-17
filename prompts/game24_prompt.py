from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solvers.game24_tot.state import State

BASELINE_PROMPT = """You are solving one Game of 24 puzzle.
Use numbers and basic arithmetic operations (+ - * /) to obtain 24.

Examples:
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24

Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24

Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24

Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24

Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24

Rules:
1. Follow the same format as the examples above.
2. End with: Answer: <final expression> = 24
3. Use each number exactly once. Only use +, -, *, /, and parentheses.
4. No extra explanation. No markdown. No code blocks.

Now solve this puzzle:
Input: {input_text}
Answer:"""


COT_PROMPT = """You are solving the Game of 24 puzzle.
Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.

Examples:
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Rules:
1. Follow the same format as the examples above.
2. End with: Answer: <final expression> = 24
The expression contains each number exactly once.
3. Use each number exactly once. Only use +, -, *, /, and parentheses.
4. No extra explanation. No markdown. No code blocks.

Now solve this puzzle:
Input: {input_text}
Steps:"""


def build_baseline_prompt(input_text: str) -> str:
    """Insert one puzzle into the baseline prompt."""
    return BASELINE_PROMPT.format(
        input_text=input_text,
    )


def build_cot_prompt(input_text: str) -> str:
    """Insert one puzzle into the CoT prompt."""
    return COT_PROMPT.format(
        input_text=input_text,
    )

def build_value_prompt(
    states: list[State],
) -> str:
    """
    构造 Qwen 的状态评价 prompt。

    候选操作由程序生成。
    Qwen 不生成操作，只评价每个状态继续得到 24 的希望。
    """
    lines = [
        (
            "Evaluate whether each remaining-number state "
            "can still reach exactly 24."
        ),
        (
            "Each state contains the remaining values "
            "after valid arithmetic operations."
        ),
        (
            "All original input numbers have been used "
            "legally to produce these values."
        ),
        (
            "Determine how promising each state is for "
            "reaching exactly 24 using only +, -, *, and /."
        ),
        "",
        "Scores:",
        "1.0 = definitely solvable",
        "0.75 = highly promising",
        "0.35 = uncertain",
        "0.0 = impossible",
        "",
        "Return JSON only.",
        "Do not include explanations.",
        "Do not use Markdown or code blocks.",
        "Return exactly one score for every state, in the same order.",
        "",
        "Return only compact JSON:",
        '{"scores":[1.0,0.75,0.35,0.0]}',
        "",
        "Examples:",
        "1. remaining values: 24; score: 1.0",
        "2. remaining values: 10 14; score: 1.0",
        "3. remaining values: 6 4; score: 1.0",
        "4. remaining values: 3 8; score: 1.0",
        "5. remaining values: 5 5; score: 0.0",
        "6. remaining values: 1 1 1; score: 0.0",
        "",
        "States to evaluate:",
    ]

    for state_id, state in enumerate(
        states,
        start=1,
    ):
        lines.append(
            (
                f"{state_id}. "
                f"remaining values: "
                f"{state.numbers_text()}; "
                f"source expressions: "
                f"{state.expressions_text()}"
            )
        )

    return "\n".join(lines)
