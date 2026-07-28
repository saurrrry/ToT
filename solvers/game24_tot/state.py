"""State objects shared by Game24 search algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable


def format_fraction(value: Fraction) -> str:
    """Render integer fractions as integers and others as a/b."""
    if value.denominator == 1:
        return str(value.numerator)

    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class Term:
    """One remaining value and the expression that produced it."""

    value: Fraction
    expression: str

    def to_dict(self) -> dict[str, str]:
        return {
            "value": format_fraction(self.value),
            "expression": self.expression,
        }


@dataclass(frozen=True)
class State:
    """A Game24 search state with remaining terms and history."""

    terms: tuple[Term, ...]

    steps: tuple[str, ...] = field(
        default_factory=tuple,
    )

    @classmethod
    def initial(
        cls,
        numbers: list[int],
    ) -> State:
        """Create the initial state from four input numbers."""
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
        """Return the number of operations already applied."""
        return 4 - len(self.terms)

    @property
    def key(self) -> tuple[tuple[int, int], ...]:
        """Deduplicate states by remaining values, not expression text."""
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
        """Return whether this state has reached the target value."""
        return (
            len(self.terms) == 1
            and self.terms[0].value
            == Fraction(target, 1)
        )

    def is_terminal(self) -> bool:
        """Return whether no further binary operation can be applied."""
        return len(self.terms) == 1

    def solution_expression(self) -> str | None:
        """Return the expression when the state has a single term."""
        if len(self.terms) != 1:
            return None

        return self.terms[0].expression

    def numbers_text(self) -> str:
        """Format remaining values for value prompts."""
        return " ".join(
            format_fraction(term.value)
            for term in self.terms
        )

    def expressions_text(self) -> str:
        """Format each remaining value with its source expression."""
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
    """Common result returned by Game24 search algorithms."""

    solution: State | None

    expanded_nodes: int = 0
    generated_nodes: int = 0

    # Key search events saved to experiment JSON.
    trace: list[dict] = field(
        default_factory=list,
    )

    @property
    def solved(self) -> bool:
        return (
            self.solution is not None
            and self.solution.is_goal()
        )


# Search algorithms call this to score states in input order.
StateScorer = Callable[
    [list[State]],
    list[float],
]


def _sort_terms(
    terms: tuple[Term, ...] | list[Term],
) -> tuple[Term, ...]:
    """Sort terms for stable display, hashing, and reproducible search."""
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
    """Create a normalized state."""
    return State(
        terms=_sort_terms(terms),
        steps=steps,
    )
