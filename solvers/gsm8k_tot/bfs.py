from __future__ import annotations

from collections.abc import Callable

from solvers.gsm8k_tot.state import (
    GSM8KSearchResult,
    GSM8KState,
)


StateGenerator = Callable[
    [GSM8KState],
    list[GSM8KState],
]

StateScorer = Callable[
    [list[GSM8KState]],
    list[float],
]


def bfs_search(
    initial_state: GSM8KState,
    generator: StateGenerator,
    scorer: StateScorer,
    *,
    beam_width: int = 3,
    max_depth: int = 6,
    max_expanded_nodes: int = 100,
) -> GSM8KSearchResult:
    """
    Run breadth-first Tree of Thoughts with beam pruning.

    Each level expands the current frontier, scores new candidates, records
    terminal candidates, and keeps the top non-terminal states for expansion.
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be greater than 0")

    if max_depth <= 0:
        raise ValueError("max_depth must be greater than 0")

    if max_expanded_nodes <= 0:
        raise ValueError("max_expanded_nodes must be greater than 0")

    if initial_state.is_terminal():
        return GSM8KSearchResult(
            solution=initial_state,
            expanded_nodes=0,
            generated_nodes=0,
            trace=[
                {
                    "event": "initial_state_terminal",
                    "state": initial_state.to_dict(),
                }
            ],
        )

    frontier = [initial_state]

    expanded_nodes = 0
    generated_nodes = 0
    trace: list[dict] = []
    best_terminal: tuple[float, GSM8KState] | None = None
    budget_exhausted = False

    for depth in range(max_depth):
        candidates: list[GSM8KState] = []
        seen_candidate_steps: set[str] = set()
        duplicate_candidate_steps = 0

        for state in frontier:
            if expanded_nodes >= max_expanded_nodes:
                budget_exhausted = True
                break

            if state.is_terminal():
                continue

            expanded_nodes += 1

            children = generator(state)
            generated_nodes += len(children)

            for child in children:
                step_key = _candidate_step_key(child)

                if step_key in seen_candidate_steps:
                    duplicate_candidate_steps += 1
                    continue

                seen_candidate_steps.add(step_key)
                candidates.append(child)

        if not candidates:
            trace.append(
                {
                    "event": "gsm8k_bfs_level",
                    "depth": depth + 1,
                    "candidate_count": 0,
                    "duplicate_candidate_steps": duplicate_candidate_steps,
                    "expanded_nodes": expanded_nodes,
                    "generated_nodes": generated_nodes,
                    "budget_exhausted": budget_exhausted,
                    "selected": [],
                }
            )
            break

        scores = scorer(candidates)

        if len(scores) != len(candidates):
            raise ValueError(
                "scorer must return exactly one score for each candidate: "
                f"got {len(scores)} scores for {len(candidates)} candidates"
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        for state, score in ranked:
            if not state.is_terminal():
                continue

            if (
                best_terminal is None
                or score > best_terminal[0]
            ):
                best_terminal = (score, state)

        selected = [
            (state, score)
            for state, score in ranked
            if not state.is_terminal()
        ][:beam_width]

        trace.append(
            {
                "event": "gsm8k_bfs_level",
                "depth": depth + 1,
                "candidate_count": len(candidates),
                "duplicate_candidate_steps": duplicate_candidate_steps,
                "terminal_count": sum(
                    1
                    for state, _ in ranked
                    if state.is_terminal()
                ),
                "expanded_nodes": expanded_nodes,
                "generated_nodes": generated_nodes,
                "budget_exhausted": budget_exhausted,
                "best_terminal": (
                    {
                        "score": best_terminal[0],
                        "state": best_terminal[1].to_dict(),
                    }
                    if best_terminal is not None
                    else None
                ),
                "selected": [
                    {
                        "score": score,
                        "state": state.to_dict(),
                    }
                    for state, score in selected
                ],
            }
        )

        frontier = [
            state
            for state, _ in selected
        ]

        if not frontier:
            break

        if budget_exhausted:
            break

    return GSM8KSearchResult(
        solution=_best_terminal_state(
            best_terminal
        ),
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        trace=trace,
    )


def _best_terminal_state(
    best_terminal: tuple[float, GSM8KState] | None,
) -> GSM8KState | None:
    if best_terminal is None:
        return None

    return best_terminal[1]


def _candidate_step_key(
    state: GSM8KState,
) -> str:
    if not state.steps:
        return ""

    return state.steps[-1].lower()
