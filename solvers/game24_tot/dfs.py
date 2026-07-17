from __future__ import annotations

from .generator import (
    generate_successors,
)
from .state import (
    SearchResult,
    State,
    StateScorer,
)


def dfs_search(
    initial_state: State,
    scorer: StateScorer,
    *,
    branch_limit: int | None = None,
    max_expanded_nodes: int = 1000,
) -> SearchResult:
    """
    模型引导的深度优先搜索。

    branch_limit:
        每个节点最多深入多少个候选。

        None 表示所有候选都可能被搜索。
    """
    if initial_state.is_goal():
        return SearchResult(
            solution=initial_state,
        )

    visited: set[
        tuple[tuple[int, int], ...]
    ] = set()

    expanded_nodes = 0
    generated_nodes = 0
    trace: list[dict] = []

    def visit(
        state: State,
    ) -> State | None:
        nonlocal expanded_nodes
        nonlocal generated_nodes

        if state.is_goal():
            return state

        if state.is_terminal():
            return None

        if expanded_nodes >= max_expanded_nodes:
            return None

        if state.key in visited:
            return None

        visited.add(state.key)
        expanded_nodes += 1

        children = generate_successors(state)
        generated_nodes += len(children)

        # 程序生成后立即检查精确目标。
        for child in children:
            if child.is_goal():
                trace.append(
                    {
                        "event": "goal",
                        "state": child.to_dict(),
                    }
                )

                return child

        unvisited_children = [
            child
            for child in children
            if child.key not in visited
        ]

        if not unvisited_children:
            return None

        scores = scorer(
            unvisited_children
        )

        ranked = sorted(
            zip(
                unvisited_children,
                scores,
                strict=True,
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        if branch_limit is not None:
            ranked = ranked[:branch_limit]

        trace.append(
            {
                "event": "dfs_expand",
                "state": state.to_dict(),
                "children": [
                    {
                        "score": score,
                        "state": child.to_dict(),
                    }
                    for child, score in ranked
                ],
            }
        )

        for child, _ in ranked:
            solution = visit(child)

            if solution is not None:
                return solution

        return None

    solution = visit(initial_state)

    return SearchResult(
        solution=solution,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        trace=trace,
    )
