from __future__ import annotations

from pathlib import Path


# 当前文件所在目录
PROJECT_ROOT = Path(__file__).resolve().parent

# Game of 24 数据集默认路径。
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "game24.jsonl"
)

# 实验结果保存目录。
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# 默认随机种子。
DEFAULT_RANDOM_SEED = 42

# 默认只测试 10 道题。
DEFAULT_LIMIT = 10

# 默认模型名称。
DEFAULT_MODEL_NAME = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# 模型生成参数。

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512

# 请求 Ollama 时的最大等待时间，单位为秒。
DEFAULT_REQUEST_TIMEOUT = 300

# ============================================================
# Game24 ToT search configuration
# ============================================================

DEFAULT_TOT_STRATEGY = "bfs"

# Qwen 每次最多评价多少个状态。
DEFAULT_VALUE_BATCH_SIZE = 12

# BFS 每一层保留多少个状态。
DEFAULT_BEAM_WIDTH = 3

# DFS 每个节点最多深入多少个候选。
#
# None 表示不限制。
DEFAULT_DFS_BRANCH_LIMIT = None

# A* 中模型启发式分数的权重。
DEFAULT_ASTAR_HEURISTIC_WEIGHT = 2.0

# MCTS 迭代次数。
DEFAULT_MCTS_ITERATIONS = 100

# MCTS UCT 探索系数。
DEFAULT_MCTS_EXPLORATION_WEIGHT = 1.4

# 每道题最多扩展多少个节点。
DEFAULT_MAX_EXPANDED_NODES = 100
