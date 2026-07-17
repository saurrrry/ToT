from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable


def format_fraction(value: Fraction) -> str:
    """
    把 Fraction 转换成适合显示的字符串。

    例如：
        Fraction(4, 1)  -> "4"
        Fraction(8, 3)  -> "8/3"
    """
    if value.denominator == 1:
        return str(value.numerator)

    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class Term:
    """
    一个中间计算项。

    value:
        该项的精确数值。

    expression:
        产生该数值的完整表达式。

    例如：

        Term(
            value=Fraction(12, 1),
            expression="(4 + 8)",
        )
    """

    value: Fraction
    expression: str

    def to_dict(self) -> dict[str, str]:
        return {
            "value": format_fraction(self.value),
            "expression": self.expression,
        }


@dataclass(frozen=True)
class State:
    """
    Game of 24 搜索中的一个状态。

    terms:
        当前剩余的数字及其对应表达式。

    steps:
        从初始状态到当前状态所执行的运算步骤。

    例如：

        输入：
            4 4 6 8

        执行：
            4 + 8 = 12

        当前状态：
            4 6 12

        terms 中保存：
            4
            6
            (4 + 8)
    """

    terms: tuple[Term, ...]

    steps: tuple[str, ...] = field(
        default_factory=tuple,
    )

    @classmethod
    def initial(
        cls,
        numbers: list[int],
    ) -> State:
        """
        从四个输入数字创建初始状态。
        """
        terms = tuple(
            Term(
                value=Fraction(number, 1),
                expression=str(number),
            )
            for number in numbers
        )

        return cls(
            terms=_sort_terms(terms),
            steps=(),
        )

    @property
    def depth(self) -> int:
        """
        当前搜索深度。

        初始状态有 4 个数字，深度为 0。
        每次操作减少一个数字，因此：

            4 个数字 -> depth 0
            3 个数字 -> depth 1
            2 个数字 -> depth 2
            1 个数字 -> depth 3
        """
        return 4 - len(self.terms)

    @property
    def key(self) -> tuple[tuple[int, int], ...]:
        """
        状态去重所使用的键。

        只比较当前剩余数值，不比较表达式字符串。

        例如下面两个状态在后续搜索上等价：

            (4 + 8), 6, 4
            (8 + 4), 6, 4

        因此它们使用同一个 key。
        """
        values = sorted(
            (
                term.value.numerator,
                term.value.denominator,
            )
            for term in self.terms
        )

        return tuple(values)

    def is_goal(
        self,
        target: int = 24,
    ) -> bool:
        """
        判断当前状态是否已经得到目标值。
        """
        return (
            len(self.terms) == 1
            and self.terms[0].value
            == Fraction(target, 1)
        )

    def is_terminal(self) -> bool:
        """
        只剩一个数时，无法继续执行二元运算。
        """
        return len(self.terms) == 1

    def solution_expression(self) -> str | None:
        """
        如果当前状态只剩一个项，返回它的完整表达式。
        """
        if len(self.terms) != 1:
            return None

        return self.terms[0].expression

    def numbers_text(self) -> str:
        """
        返回给模型评价的剩余数值。

        例如：
            "4 6 12"
            "1/3 8"
        """
        return " ".join(
            format_fraction(term.value)
            for term in self.terms
        )

    def expressions_text(self) -> str:
        """
        返回当前每个值的来源表达式。
        """
        return " | ".join(
            (
                f"{term.expression}"
                f"={format_fraction(term.value)}"
            )
            for term in self.terms
        )

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "numbers": [
                format_fraction(term.value)
                for term in self.terms
            ],
            "terms": [
                term.to_dict()
                for term in self.terms
            ],
            "steps": list(self.steps),
            "is_goal": self.is_goal(),
        }


@dataclass
class SearchResult:
    """
    四种搜索算法统一返回的结果。
    """

    solution: State | None

    expanded_nodes: int = 0
    generated_nodes: int = 0

    # 保存搜索过程中的关键信息，
    # 后续会写入实验 JSON。
    trace: list[dict] = field(
        default_factory=list,
    )

    @property
    def solved(self) -> bool:
        return (
            self.solution is not None
            and self.solution.is_goal()
        )


# 搜索算法通过这个函数类型调用模型评价。
#
# 输入：
#     一批 State
#
# 输出：
#     与输入顺序对应的 float 分数
#
# 分数范围约定为 0 到 1。
StateScorer = Callable[
    [list[State]],
    list[float],
]


def _sort_terms(
    terms: tuple[Term, ...] | list[Term],
) -> tuple[Term, ...]:
    """
    对状态中的项进行稳定排序。

    排序不会影响数学结果，但可以：

    1. 让状态显示顺序稳定；
    2. 提高实验可复现性；
    3. 减少搜索顺序受 Python 容器影响。
    """
    return tuple(
        sorted(
            terms,
            key=lambda term: (
                term.value,
                term.expression,
            ),
        )
    )


def create_state(
    terms: list[Term],
    steps: tuple[str, ...],
) -> State:
    """
    创建并规范化一个新状态。
    """
    return State(
        terms=_sort_terms(terms),
        steps=steps,
    )