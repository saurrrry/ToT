# ToT

Tree-of-Thoughts style experiments for the Game of 24, wrapped around a Qwen model served by Ollama.

## Project Goal

This project compares direct prompting, chain-of-thought prompting, and search-based ToT variants on Game24.

## Directory Structure

```text
data/                 Game24 data
datasets/             Dataset loaders
evaluation/           Evaluation and result writing
models/               Model backend interfaces and Ollama backend
prompts/              Baseline, CoT and value-evaluation prompts
solvers/              Baseline, CoT and ToT solvers
solvers/game24_tot/   BFS, DFS, A*-style, MCTS-style search
utils/                Parsing and result IO helpers
verifier/             Exact Game24 expression verifier
tests/                Unit tests
```

## Environment

Install Python dependencies:

```powershell
pip install requests pytest
```

Install Ollama, then pull a Qwen model:

```powershell
ollama pull qwen2.5:7b
```

For faster local testing:

```powershell
ollama pull qwen2.5:3b
```

## Data Format

`data/game24.jsonl` contains one JSON object per line with:

```json
{
  "id": "official-0001",
  "numbers": [1, 7, 8, 11],
  "solutions": ["..."],
  "solvable": true
}
```

Extra fields are preserved as `dataset_metadata` in result files.

## Run

Run from the workspace root `D:\files\NJU`:

```powershell
python -m ToT.Tot_mcts.main --method baseline --limit 10
python -m ToT.Tot_mcts.main --method cot --limit 10
python -m ToT.Tot_mcts.main --method tot_bfs --limit 10
python -m ToT.Tot_mcts.main --method tot_dfs --limit 10
python -m ToT.Tot_mcts.main --method tot_astar --limit 10
python -m ToT.Tot_mcts.main --method tot_mcts --limit 10
```

Run every method on the same sampled problems:

```powershell
python -m ToT.Tot_mcts.main --method all --limit 10
```

## Important Parameters

```text
--model                 Ollama model name
--num-ctx               Ollama context window
--baseline-max-tokens   Baseline generation budget
--cot-max-tokens        CoT generation budget
--value-max-tokens      ToT value-scoring generation budget
--beam-width            Beam width for ToT Beam BFS
--dfs-branch-limit      Maximum ranked children explored by DFS
--astar-weight          Weight for the model heuristic in A*-style search
--mcts-iterations       MCTS iteration budget
--mcts-exploration      UCT exploration constant
--value-batch-size      Number of states scored per model call
--max-expanded-nodes    Per-puzzle search expansion cap
```

## Algorithm Notes

`tot_bfs` is Beam BFS, not exhaustive BFS.

`tot_astar` is an LLM-guided weighted A*-style search. The model score is not guaranteed to be an admissible heuristic, so classical A* optimality is not guaranteed.

`tot_mcts` is value-based MCTS. Leaf rewards are derived from Qwen state scores rather than full random rollouts.

## Results

Results are written under:

```text
results/game24/
```

The repository ignores `results/` because result JSON files can include large model traces.

## Tests

Run:

```powershell
pytest
```
