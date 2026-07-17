from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_ASTAR_HEURISTIC_WEIGHT,
    DEFAULT_BASELINE_MAX_TOKENS,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_COT_MAX_TOKENS,
    DEFAULT_DATA_PATH,
    DEFAULT_DFS_BRANCH_LIMIT,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_LIMIT,
    DEFAULT_MAX_EXPANDED_NODES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MCTS_EXPLORATION_WEIGHT,
    DEFAULT_MCTS_ITERATIONS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEMPERATURE,
    DEFAULT_VALUE_BATCH_SIZE,
    DEFAULT_VALUE_MAX_TOKENS,
)
from datasets.game24 import (
    Game24Sample,
    load_game24_dataset,
)
from evaluation.game24_evaluator import (
    evaluate_game24,
)
from models.ollama_model import OllamaModel
from solvers.base_solver import BaseSolver
from solvers.baseline import BaselineSolver
from solvers.cot import CoTSolver
from solvers.game24_tot import (
    Game24ToTSolver,
)


# 命令行中的 ToT 方法名与内部 strategy 的映射。
TOT_METHODS = {
    "tot_bfs": "bfs",
    "tot_dfs": "dfs",
    "tot_astar": "astar",
    "tot_mcts": "mcts",
}


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline, CoT and "
            "search-based ToT methods "
            "on the Game of 24 dataset."
        )
    )

    parser.add_argument(
        "--method",
        choices=[
            "baseline",
            "cot",
            "tot_bfs",
            "tot_dfs",
            "tot_astar",
            "tot_mcts",
            "all",
        ],
        default="baseline",
        help=(
            "Method to evaluate. "
            "'all' runs baseline, cot, bfs, dfs, "
            "astar and mcts on the same samples."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Number of dataset samples to evaluate. "
            f"Default: {DEFAULT_LIMIT}."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed used for dataset shuffling, "
            "model generation and MCTS."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Ollama model name. "
            f"Default: {DEFAULT_MODEL_NAME}."
        ),
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the Game24 JSONL dataset.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Root directory used to save "
            "experiment JSON files."
        ),
    )

    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama server base URL.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Model sampling temperature. "
            "Use 0.0 for deterministic evaluation."
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum number of tokens generated "
            "for one model request."
        ),
    )

    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Ollama context window size in tokens.",
    )

    parser.add_argument(
        "--baseline-max-tokens",
        type=int,
        default=DEFAULT_BASELINE_MAX_TOKENS,
        help="Maximum generated tokens for baseline.",
    )

    parser.add_argument(
        "--cot-max-tokens",
        type=int,
        default=DEFAULT_COT_MAX_TOKENS,
        help="Maximum generated tokens for CoT.",
    )

    parser.add_argument(
        "--value-max-tokens",
        type=int,
        default=DEFAULT_VALUE_MAX_TOKENS,
        help="Maximum generated tokens for ToT value scoring.",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help=(
            "Do not shuffle the dataset before "
            "applying --limit."
        ),
    )

    # ========================================================
    # ToT-BFS 参数
    # ========================================================

    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help=(
            "Number of states retained at each "
            "BFS depth."
        ),
    )

    # ========================================================
    # ToT-DFS 参数
    # ========================================================

    parser.add_argument(
        "--dfs-branch-limit",
        type=int,
        default=DEFAULT_DFS_BRANCH_LIMIT,
        help=(
            "Maximum number of ranked children "
            "explored per DFS state. "
            "Omit to explore all children."
        ),
    )

    # ========================================================
    # ToT-A* 参数
    # ========================================================

    parser.add_argument(
        "--astar-weight",
        type=float,
        default=DEFAULT_ASTAR_HEURISTIC_WEIGHT,
        help=(
            "Weight of the model-based heuristic "
            "used by A*."
        ),
    )

    # ========================================================
    # ToT-MCTS 参数
    # ========================================================

    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=DEFAULT_MCTS_ITERATIONS,
        help=(
            "Maximum number of MCTS iterations "
            "per puzzle."
        ),
    )

    parser.add_argument(
        "--mcts-exploration",
        type=float,
        default=DEFAULT_MCTS_EXPLORATION_WEIGHT,
        help=(
            "UCT exploration constant used by MCTS."
        ),
    )

    # ========================================================
    # 所有搜索方法共用参数
    # ========================================================

    parser.add_argument(
        "--value-batch-size",
        type=int,
        default=DEFAULT_VALUE_BATCH_SIZE,
        help=(
            "Maximum number of states evaluated "
            "in one Qwen request."
        ),
    )

    parser.add_argument(
        "--max-expanded-nodes",
        type=int,
        default=DEFAULT_MAX_EXPANDED_NODES,
        help=(
            "Maximum number of expanded nodes "
            "for one puzzle."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    程序主入口。
    """
    args = parse_args()

    _validate_args(args)

    # 数据集只加载一次。
    #
    # 当 --method all 时，所有方法都会复用同一个 samples 列表，
    # 从而保证测试题目和顺序完全一致。
    samples = load_game24_dataset(
        args.data_path,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        limit=args.limit,
        solvable_only=True,
    )

    if not samples:
        raise RuntimeError(
            "No solvable samples were loaded "
            "from the dataset."
        )

    # 所有方法共用同一个模型后端。
    #
    # solver 会在每次调用 generate() 时传入自己的参数。
    model = OllamaModel(
        model_name=args.model,
        base_url=args.ollama_url,
        default_temperature=args.temperature,
        default_max_tokens=args.cot_max_tokens,
        default_context_length=args.num_ctx,
        keep_alive=DEFAULT_KEEP_ALIVE,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )

    common_run_config = {
        "data_path": str(args.data_path),
        "results_dir": str(args.results_dir),
        "shuffle": not args.no_shuffle,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "baseline_max_tokens": args.baseline_max_tokens,
        "cot_max_tokens": args.cot_max_tokens,
        "value_max_tokens": args.value_max_tokens,
        "keep_alive": DEFAULT_KEEP_ALIVE,
        "ollama_url": args.ollama_url,
        "limit": args.limit,
        "seed": args.seed,
    }

    methods_to_run = _resolve_methods(
        args.method
    )

    for run_index, method_name in enumerate(
        methods_to_run,
        start=1,
    ):
        # 多个方法之间打印空行。
        if run_index > 1:
            print()

        solver = _create_solver(
            method_name=method_name,
            model=model,
            args=args,
        )

        run_config = {
            **common_run_config,
            **_method_config(
                method_name,
                args,
            ),
        }

        _print_run_header(
            method_name=method_name,
            model_name=args.model,
            samples=samples,
            seed=args.seed,
        )

        evaluate_game24(
            samples=samples,
            solver=solver,
            model_name=args.model,
            seed=args.seed,
            results_dir=args.results_dir,
            run_config=run_config,
        )


def _resolve_methods(
    selected_method: str,
) -> list[str]:
    """
    将 --method 转换为实际运行的方法列表。
    """
    if selected_method == "all":
        return [
            "baseline",
            "cot",
            "tot_bfs",
            "tot_dfs",
            "tot_astar",
            "tot_mcts",
        ]

    return [selected_method]


def _create_solver(
    *,
    method_name: str,
    model: OllamaModel,
    args: argparse.Namespace,
) -> BaseSolver:
    """
    根据方法名创建 solver。
    """
    if method_name == "baseline":
        return BaselineSolver(
            model,
            temperature=args.temperature,

            # Baseline 只输出一行表达式，
            # 不需要与 CoT 相同的长输出。
            max_tokens=args.baseline_max_tokens,
        )

    if method_name == "cot":
        return CoTSolver(
            model,
            temperature=args.temperature,
            max_tokens=args.cot_max_tokens,
        )

    if method_name in TOT_METHODS:
        strategy = TOT_METHODS[method_name]

        return Game24ToTSolver(
            model,
            strategy=strategy,
            temperature=args.temperature,
            value_max_tokens=args.value_max_tokens,
            value_batch_size=(
                args.value_batch_size
            ),
            beam_width=args.beam_width,
            dfs_branch_limit=(
                args.dfs_branch_limit
            ),
            astar_heuristic_weight=(
                args.astar_weight
            ),
            mcts_iterations=(
                args.mcts_iterations
            ),
            mcts_exploration_weight=(
                args.mcts_exploration
            ),
            max_expanded_nodes=(
                args.max_expanded_nodes
            ),
        )

    raise ValueError(
        f"Unsupported method: {method_name}"
    )


def _method_config(
    method_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    生成需要保存到结果 JSON 中的方法参数。
    """
    if method_name == "baseline":
        return {
            "method": "baseline",
            "shots": 5,
            "reasoning": False,
        }

    if method_name == "cot":
        return {
            "method": "cot",
            "shots": 5,
            "reasoning": True,
        }

    if method_name == "tot_bfs":
        return {
            "method": "tot_bfs",
            "strategy": "bfs",
            "beam_width": args.beam_width,
            "value_batch_size": (
                args.value_batch_size
            ),
            "max_expanded_nodes": (
                args.max_expanded_nodes
            ),
        }

    if method_name == "tot_dfs":
        return {
            "method": "tot_dfs",
            "strategy": "dfs",
            "dfs_branch_limit": (
                args.dfs_branch_limit
            ),
            "value_batch_size": (
                args.value_batch_size
            ),
            "max_expanded_nodes": (
                args.max_expanded_nodes
            ),
        }

    if method_name == "tot_astar":
        return {
            "method": "tot_astar",
            "strategy": "astar",
            "astar_heuristic_weight": (
                args.astar_weight
            ),
            "value_batch_size": (
                args.value_batch_size
            ),
            "max_expanded_nodes": (
                args.max_expanded_nodes
            ),
        }

    if method_name == "tot_mcts":
        return {
            "method": "tot_mcts",
            "strategy": "mcts",
            "mcts_iterations": (
                args.mcts_iterations
            ),
            "mcts_exploration_weight": (
                args.mcts_exploration
            ),
            "value_batch_size": (
                args.value_batch_size
            ),
            "max_expanded_nodes": (
                args.max_expanded_nodes
            ),
        }

    return {
        "method": method_name,
    }


def _print_run_header(
    *,
    method_name: str,
    model_name: str,
    samples: list[Game24Sample],
    seed: int,
) -> None:
    """
    打印一次实验的基本信息。
    """
    print("=" * 72)
    print(f"Running method: {method_name}")
    print(f"Model: {model_name}")
    print(f"Samples: {len(samples)}")
    print(f"Seed: {seed}")
    print("=" * 72)


def _validate_args(
    args: argparse.Namespace,
) -> None:
    """
    在开始加载模型和数据之前检查命令行参数。
    """
    if args.limit < 0:
        raise ValueError(
            "--limit must be non-negative"
        )

    if args.max_tokens <= 0:
        raise ValueError(
            "--max-tokens must be positive"
        )

    if args.num_ctx <= 0:
        raise ValueError(
            "--num-ctx must be positive"
        )

    if args.baseline_max_tokens <= 0:
        raise ValueError(
            "--baseline-max-tokens must be positive"
        )

    if args.cot_max_tokens <= 0:
        raise ValueError(
            "--cot-max-tokens must be positive"
        )

    if args.value_max_tokens <= 0:
        raise ValueError(
            "--value-max-tokens must be positive"
        )

    if args.temperature < 0:
        raise ValueError(
            "--temperature must be non-negative"
        )

    if args.beam_width <= 0:
        raise ValueError(
            "--beam-width must be positive"
        )

    if (
        args.dfs_branch_limit is not None
        and args.dfs_branch_limit <= 0
    ):
        raise ValueError(
            "--dfs-branch-limit must be positive"
        )

    if args.astar_weight < 0:
        raise ValueError(
            "--astar-weight must be non-negative"
        )

    if args.mcts_iterations <= 0:
        raise ValueError(
            "--mcts-iterations must be positive"
        )

    if args.mcts_exploration < 0:
        raise ValueError(
            "--mcts-exploration must be non-negative"
        )

    if args.value_batch_size <= 0:
        raise ValueError(
            "--value-batch-size must be positive"
        )

    if args.max_expanded_nodes <= 0:
        raise ValueError(
            "--max-expanded-nodes must be positive"
        )


if __name__ == "__main__":
    main()
