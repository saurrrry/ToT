from __future__ import annotations

import heapq
from itertools import count

from solvers.game24_tot.generator import (
    generate_successors,
)
from solvers.game24_tot.state import (
    SearchResult,
    State,
    StateScorer,
)


def astar_search(
    initial_state: State,
    scorer: StateScorer,
    *,
    heuristic_weight: float = 2.0,
    max_expanded_nodes: int = 1000,
) -> SearchResult:
    """
    LLM-guided weighted A*-style search.

    The model-derived heuristic is not guaranteed to be admissible,
    so this implementation does not guarantee classical A* optimality.

    优先级：

        f(n) = g(n) + weight * (1 - score)

    其中：

        g(n):
            当前搜索深度。

        score:
            Qwen 对状态的评价，范围 0 到 1。

        1 - score:
            越有希望的状态，启发式代价越低。
    """
    if initial_state.is_goal():
        return SearchResult(
            solution=initial_state,
        )

    tie_breaker = count()

    # heap 中保存：
    #
    #     priority
    #     insertion_order
    #     state
    open_heap: list[
        tuple[
            float,
            int,
            State,
        ]
    ] = []

    heapq.heappush(
        open_heap,
        (
            0.0,
            next(tie_breaker),
            initial_state,
        ),
    )

    # 保存每个状态目前已知的最小 g 值。
    best_cost: dict[
        tuple[tuple[int, int], ...],
        int,
    ] = {
        initial_state.key: 0,
    }

    expanded_nodes = 0
    generated_nodes = 0
    trace: list[dict] = []

    while open_heap:
        if expanded_nodes >= max_expanded_nodes:
            break

        priority, _, state = heapq.heappop(
            open_heap
        )

        if state.is_goal():
            return SearchResult(
                solution=state,
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                trace=trace,
            )

        if state.is_terminal():
            continue

        expanded_nodes += 1

        children = generate_successors(state)
        generated_nodes += len(children)

        for child in children:
            if child.is_goal():
                trace.append(
                    {
                        "event": "goal",
                        "priority": priority,
                        "state": child.to_dict(),
                    }
                )

                return SearchResult(
                    solution=child,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    trace=trace,
                )

        valid_children: list[State] = []

        for child in children:
            new_g = child.depth

            old_g = best_cost.get(
                child.key
            )

            if (
                old_g is not None
                and new_g >= old_g
            ):
                continue

            best_cost[child.key] = new_g
            valid_children.append(child)

        if not valid_children:
            continue

        scores = scorer(valid_children)

        expanded_record: list[dict] = []

        for child, score in zip(
            valid_children,
            scores,
            strict=True,
        ):
            g_cost = child.depth
            heuristic_cost = 1.0 - score

            child_priority = (
                g_cost
                + heuristic_weight
                * heuristic_cost
            )

            heapq.heappush(
                open_heap,
                (
                    child_priority,
                    next(tie_breaker),
                    child,
                ),
            )

            expanded_record.append(
                {
                    "g": g_cost,
                    "score": score,
                    "priority": child_priority,
                    "state": child.to_dict(),
                }
            )

        trace.append(
            {
                "event": "astar_expand",
                "parent": state.to_dict(),
                "children": expanded_record,
            }
        )

    return SearchResult(
        solution=None,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        trace=trace,
    )
