"""State objects shared by GSM8K ToT search."""

from __future__ import annotations

from dataclasses import dataclass, field

from verifier.gsm8k import extract_gsm8k_final_answer


@dataclass(frozen=True)
class GSM8KState:
    """A partial GSM8K reasoning path."""

    question: str
    steps: tuple[str, ...] = field(default_factory=tuple)
    final_answer: str | None = None

    @classmethod
    def initial(
        cls,
        question: str,
    ) -> "GSM8KState":
        """Create an empty reasoning state for one question."""
        return cls(question=question)

    @property
    def depth(self) -> int:
        """Return the number of reasoning steps."""
        return len(self.steps)

    @property
    def key(self) -> tuple[str, ...]:
        """Normalize steps for deduplication."""
        return tuple(
            _normalize_step(step)
            for step in self.steps
        )

    def is_terminal(self) -> bool:
        """Return whether the state already contains a final answer."""
        return self.final_answer is not None

    def add_step(
        self,
        step: str,
    ) -> "GSM8KState":
        """Return a new state with one cleaned reasoning step appended."""
        cleaned_step = " ".join(step.strip().split())
        final_answer = extract_gsm8k_final_answer(
            cleaned_step
        )

        return GSM8KState(
            question=self.question,
            steps=(
                *self.steps,
                cleaned_step,
            ),
            final_answer=final_answer,
        )

    def solution_expression(self) -> str | None:
        """Return the final answer when present."""
        return self.final_answer

    def to_dict(self) -> dict:
        """Serialize the state for traces and result files."""
        return {
            "question": self.question,
            "steps": list(self.steps),
            "depth": self.depth,
            "final_answer": self.final_answer,
            "is_terminal": self.is_terminal(),
        }


@dataclass
class GSM8KSearchResult:
    """Search result for GSM8K ToT."""

    solution: GSM8KState | None
    expanded_nodes: int = 0
    generated_nodes: int = 0
    trace: list[dict] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return (
            self.solution is not None
            and self.solution.final_answer is not None
        )


def _normalize_step(
    step: str,
) -> str:
    """Normalize a reasoning step for equality checks."""
    return " ".join(step.lower().split())
