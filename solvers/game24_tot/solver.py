from __future__ import annotations

import json
from typing import Literal

from ...datasets.game24 import Game24Sample
from ...models.base_model import BaseModel
from ...prompts.game24_prompt import (
    build_value_prompt,
)
from ..base_solver import (
    BaseSolver,
    SolverResult,
)
from .astar import (
    astar_search,
)
from .bfs import (
    bfs_search,
)
from .dfs import (
    dfs_search,
)
from .mcts import (
    mcts_search,
)
from .state import (
    State,
)
from ...utils.parsing import parse_state_scores


SearchStrategy = Literal[
    "bfs",
    "dfs",
    "astar",
    "mcts",
]


RATING_SCORES = {
    "sure": 1.0,
    "likely": 0.75,
    "maybe": 0.35,
    "impossible": 0.0,
}


class StateValueEvaluator:
    """
    使用 Qwen 对中间状态进行批量评价。

    该类同时负责：

    1. prompt 构造；
    2. 模型调用；
    3. JSON 解析；
    4. 状态评价缓存；
    5. token 和时间统计。
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        batch_size: int = 20,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.seed = seed

        self.cache: dict[
            tuple[tuple[int, int], ...],
            float,
        ] = {}

        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.duration_seconds = 0.0

        self.evaluation_logs: list[dict] = []

    def score_states(
        self,
        states: list[State],
    ) -> list[float]:
        """
        按输入顺序返回每个状态的分数。
        """
        if not states:
            return []

        # 同一批中也可能存在重复状态。
        unique_uncached: list[State] = []
        pending_keys: set[
            tuple[tuple[int, int], ...]
        ] = set()

        for state in states:
            if state.is_goal():
                self.cache[state.key] = 1.0
                continue

            if state.key in self.cache:
                continue

            if state.key in pending_keys:
                continue

            pending_keys.add(state.key)
            unique_uncached.append(state)

        for start_index in range(
            0,
            len(unique_uncached),
            self.batch_size,
        ):
            batch = unique_uncached[
                start_index:
                start_index + self.batch_size
            ]

            self._evaluate_batch(batch)

        return [
            self.cache.get(
                state.key,
                RATING_SCORES["maybe"],
            )
            for state in states
        ]

    def _evaluate_batch(
        self,
        states: list[State],
    ) -> None:
        prompt = build_value_prompt(states)

        generation_seed = (
            self.seed
            + self.model_calls
        )

        generation = self.model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=generation_seed,
        )

        self.model_calls += 1

        if generation.prompt_tokens is not None:
            self.prompt_tokens += (
                generation.prompt_tokens
            )

        if generation.completion_tokens is not None:
            self.completion_tokens += (
                generation.completion_tokens
            )

        if generation.duration_seconds is not None:
            self.duration_seconds += (
                generation.duration_seconds
            )

        scores = parse_state_scores(
            generation.text,
            expected_count=len(states),
        )

        batch_log = {
            "prompt": prompt,
            "raw_response": generation.text,
            "ratings": [],
        }

        for state, score in zip(
            states,
            scores,
            strict=True,
        ):
            self.cache[state.key] = score

            batch_log["ratings"].append(
                {
                    "state": state.to_dict(),
                    "score": score,
                }
            )

        self.evaluation_logs.append(
            batch_log
        )


class Game24ToTSolver(BaseSolver):
    """
    Game of 24 的统一搜索 solver。

    候选节点由程序生成，
    Qwen 只负责评价节点。
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        strategy: SearchStrategy = "bfs",
        temperature: float = 0.0,
        max_tokens: int = 512,
        value_batch_size: int = 20,
        beam_width: int = 10,
        dfs_branch_limit: int | None = None,
        astar_heuristic_weight: float = 2.0,
        mcts_iterations: int = 100,
        mcts_exploration_weight: float = 1.4,
        max_expanded_nodes: int = 1000,
    ) -> None:
        self.model = model
        self.strategy = strategy

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.value_batch_size = value_batch_size

        self.beam_width = beam_width
        self.dfs_branch_limit = dfs_branch_limit

        self.astar_heuristic_weight = (
            astar_heuristic_weight
        )

        self.mcts_iterations = (
            mcts_iterations
        )

        self.mcts_exploration_weight = (
            mcts_exploration_weight
        )

        self.max_expanded_nodes = (
            max_expanded_nodes
        )

    @property
    def name(self) -> str:
        """
        保存结果时使用不同方法名。

        例如：
            tot_bfs
            tot_dfs
            tot_astar
            tot_mcts
        """
        return f"tot_{self.strategy}"

    def solve(
        self,
        sample: Game24Sample,
        *,
        seed: int | None = None,
    ) -> SolverResult:
        selected_seed = (
            42
            if seed is None
            else seed
        )

        initial_state = State.initial(
            sample.numbers
        )

        value_evaluator = StateValueEvaluator(
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            batch_size=self.value_batch_size,
            seed=selected_seed,
        )

        if self.strategy == "bfs":
            search_result = bfs_search(
                initial_state,
                value_evaluator.score_states,
                beam_width=self.beam_width,
                max_expanded_nodes=(
                    self.max_expanded_nodes
                ),
            )

        elif self.strategy == "dfs":
            search_result = dfs_search(
                initial_state,
                value_evaluator.score_states,
                branch_limit=(
                    self.dfs_branch_limit
                ),
                max_expanded_nodes=(
                    self.max_expanded_nodes
                ),
            )

        elif self.strategy == "astar":
            search_result = astar_search(
                initial_state,
                value_evaluator.score_states,
                heuristic_weight=(
                    self.astar_heuristic_weight
                ),
                max_expanded_nodes=(
                    self.max_expanded_nodes
                ),
            )

        elif self.strategy == "mcts":
            search_result = mcts_search(
                initial_state,
                value_evaluator.score_states,
                iterations=self.mcts_iterations,
                exploration_weight=(
                    self.mcts_exploration_weight
                ),
                random_seed=selected_seed,
                max_expanded_nodes=(
                    self.max_expanded_nodes
                ),
            )

        else:
            raise ValueError(
                f"Unsupported search strategy: "
                f"{self.strategy}"
            )

        expression = None

        if search_result.solution is not None:
            expression = (
                search_result
                .solution
                .solution_expression()
            )

        metadata = {
            "method": self.name,
            "strategy": self.strategy,
            "search": {
                "solved": search_result.solved,
                "expanded_nodes": (
                    search_result.expanded_nodes
                ),
                "generated_nodes": (
                    search_result.generated_nodes
                ),
                "solution_steps": (
                    list(
                        search_result
                        .solution
                        .steps
                    )
                    if search_result.solution
                    is not None
                    else []
                ),
                "trace": search_result.trace,
            },
            "value_evaluator": {
                "model_calls": (
                    value_evaluator.model_calls
                ),
                "cached_states": len(
                    value_evaluator.cache
                ),
                "logs": (
                    value_evaluator
                    .evaluation_logs
                ),
            },
        }

        # ToT 搜索包含多次模型调用，
        # 不存在唯一的 raw response。
        #
        # 这里保存所有评价调用的 JSON 文本，
        # 方便与现有 evaluator 接口兼容。
        raw_response = json.dumps(
            value_evaluator.evaluation_logs,
            ensure_ascii=False,
        )

        return SolverResult(
            expression=expression,
            raw_response=raw_response,

            # ToT 没有唯一 prompt。
            # 详细 prompt 已保存在 metadata logs 中。
            prompt="",

            duration_seconds=(
                value_evaluator
                .duration_seconds
            ),
            prompt_tokens=(
                value_evaluator
                .prompt_tokens
            ),
            completion_tokens=(
                value_evaluator
                .completion_tokens
            ),
            metadata=metadata,
        )
