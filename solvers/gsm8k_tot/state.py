from __future__ import annotations

from dataclasses import dataclass, field

from verifier.gsm8k import extract_gsm8k_final_answer


@dataclass(frozen=True)
class GSM8KState:
    question: str
    steps: tuple[str, ...] = field(default_factory=tuple)
    final_answer: str | None = None

    @classmethod
    def initial(
        cls,
        question: str,
    ) -> "GSM8KState":
        return cls(question=question)

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(
            _normalize_step(step)
            for step in self.steps
        )

    def is_terminal(self) -> bool:
        return self.final_answer is not None

    def add_step(
        self,
        step: str,
    ) -> "GSM8KState":
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
        return self.final_answer

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "steps": list(self.steps),
            "depth": self.depth,
            "final_answer": self.final_answer,
            "is_terminal": self.is_terminal(),
        }


@dataclass
class GSM8KSearchResult:
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
    return " ".join(step.lower().split())
