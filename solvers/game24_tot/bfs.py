from __future__ import annotations

from .generator import (
    generate_successors,
)
from .state import (
    SearchResult,
    State,
    StateScorer,
)


def bfs_search(
    initial_state: State,
    scorer: StateScorer,
    *,
    beam_width: int = 10,
    max_expanded_nodes: int = 1000,
) -> SearchResult:
    """
    基于层次扩展的 Beam BFS。

    每一层流程：

        当前 frontier
            ↓
        程序生成所有子节点
            ↓
        检查是否已经得到 24
            ↓
        模型批量评价
            ↓
        保留 top-k
    """
    if initial_state.is_goal():
        return SearchResult(
            solution=initial_state,
        )

    frontier = [initial_state]

    visited = {
        initial_state.key,
    }

    expanded_nodes = 0
    generated_nodes = 0
    trace: list[dict] = []

    while frontier:
        candidates: list[State] = []

        for state in frontier:
            if expanded_nodes >= max_expanded_nodes:
                return SearchResult(
                    solution=None,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    trace=trace,
                )

            expanded_nodes += 1

            children = generate_successors(state)
            generated_nodes += len(children)

            for child in children:
                # 最终目标优先直接返回，
                # 不需要再让模型评价。
                if child.is_goal():
                    trace.append(
                        {
                            "event": "goal",
                            "state": child.to_dict(),
                        }
                    )

                    return SearchResult(
                        solution=child,
                        expanded_nodes=expanded_nodes,
                        generated_nodes=generated_nodes,
                        trace=trace,
                    )

                if child.key in visited:
                    continue

                visited.add(child.key)
                candidates.append(child)

        if not candidates:
            break

        scores = scorer(candidates)

        ranked = sorted(
            zip(
                candidates,
                scores,
                strict=True,
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        selected = ranked[:beam_width]

        trace.append(
            {
                "event": "bfs_level",
                "depth": selected[0][0].depth
                if selected
                else None,
                "candidate_count": len(candidates),
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

    return SearchResult(
        solution=None,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        trace=trace,
    )
