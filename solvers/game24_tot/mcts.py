from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .generator import (
    generate_successors,
)
from .state import (
    SearchResult,
    State,
    StateScorer,
)


@dataclass
class MCTSNode:
    state: State

    parent: MCTSNode | None = None

    children: list[MCTSNode] = field(
        default_factory=list,
    )

    # 尚未扩展的子状态及其模型先验分数。
    untried_children: list[
        tuple[
            State,
            float,
        ]
    ] = field(
        default_factory=list,
    )

    initialized: bool = False

    visits: int = 0
    value_sum: float = 0.0

    # 模型对当前节点的先验评价。
    prior: float = 0.5

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits


def mcts_search(
    initial_state: State,
    scorer: StateScorer,
    *,
    iterations: int = 100,
    exploration_weight: float = 1.4,
    prior_weight: float = 0.25,
    random_seed: int = 42,
    max_expanded_nodes: int = 1000,
) -> SearchResult:
    """
    使用 Qwen 状态评分作为叶节点价值的 MCTS。
    """
    if initial_state.is_goal():
        return SearchResult(
            solution=initial_state,
        )

    random_generator = random.Random(
        random_seed
    )

    root = MCTSNode(
        state=initial_state,
        prior=0.5,
    )

    expanded_nodes = 0
    generated_nodes = 0
    trace: list[dict] = []

    found_solution: State | None = None

    def initialize_node(
        node: MCTSNode,
    ) -> None:
        """
        第一次访问节点时：

        1. 程序生成全部子节点；
        2. 模型批量评价；
        3. 按评分从高到低保存为未扩展节点。
        """
        nonlocal generated_nodes

        if node.initialized:
            return

        node.initialized = True

        if node.state.is_terminal():
            return

        children = generate_successors(
            node.state
        )

        generated_nodes += len(children)

        if not children:
            return

        scores = scorer(children)

        node.untried_children = sorted(
            zip(
                children,
                scores,
                strict=True,
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

    for iteration in range(1, iterations + 1):
        if expanded_nodes >= max_expanded_nodes:
            break

        node = root
        path = [node]

        # 1. Selection
        #
        # 如果节点没有未扩展子节点，
        # 就使用 UCT 选择一个已有子节点。
        while True:
            if node.state.is_goal():
                found_solution = node.state
                break

            initialize_node(node)

            if node.untried_children:
                break

            if not node.children:
                break

            node = _select_child(
                node,
                exploration_weight=exploration_weight,
                prior_weight=prior_weight,
                random_generator=random_generator,
            )

            path.append(node)

        if found_solution is not None:
            break

        # 2. Expansion
        if node.untried_children:
            child_state, child_prior = (
                node.untried_children.pop(0)
            )

            child_node = MCTSNode(
                state=child_state,
                parent=node,
                prior=child_prior,
            )

            node.children.append(child_node)

            node = child_node
            path.append(node)

            expanded_nodes += 1

        # 3. Evaluation
        if node.state.is_goal():
            reward = 1.0
            found_solution = node.state

        elif node.state.is_terminal():
            reward = 0.0

        else:
            # 节点生成时已经由模型赋予 prior，
            # 使用该 prior 作为叶节点价值。
            reward = node.prior

        # 4. Backpropagation
        for visited_node in reversed(path):
            visited_node.visits += 1
            visited_node.value_sum += reward

        trace.append(
            {
                "event": "mcts_iteration",
                "iteration": iteration,
                "reward": reward,
                "path": [
                    path_node.state.to_dict()
                    for path_node in path
                ],
            }
        )

        if found_solution is not None:
            break

    return SearchResult(
        solution=found_solution,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        trace=trace,
    )


def _select_child(
    node: MCTSNode,
    *,
    exploration_weight: float,
    prior_weight: float,
    random_generator: random.Random,
) -> MCTSNode:
    """
    使用带先验项的 UCT 选择子节点。

    score =
        exploitation
        + exploration
        + prior bonus
    """
    parent_visits = max(
        node.visits,
        1,
    )

    scored_children: list[
        tuple[
            float,
            MCTSNode,
        ]
    ] = []

    for child in node.children:
        if child.visits == 0:
            uct_score = float("inf")
        else:
            exploitation = child.mean_value

            exploration = (
                exploration_weight
                * math.sqrt(
                    math.log(parent_visits + 1)
                    / child.visits
                )
            )

            prior_bonus = (
                prior_weight
                * child.prior
                / (1 + child.visits)
            )

            uct_score = (
                exploitation
                + exploration
                + prior_bonus
            )

        scored_children.append(
            (
                uct_score,
                child,
            )
        )

    best_score = max(
        score
        for score, _ in scored_children
    )

    best_children = [
        child
        for score, child in scored_children
        if score == best_score
    ]

    return random_generator.choice(
        best_children
    )
