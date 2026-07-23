# ToT

A local Tree-of-Thoughts wrapper for LLM reasoning experiments on **Game24** and **GSM8K**.

The project compares direct input-output prompting, Chain-of-Thought prompting, Self-Consistency CoT, and search-based Tree-of-Thoughts variants using a Qwen model served by Ollama.

## Features

- **Datasets**
  - Game24: symbolic arithmetic search with exact expression verification.
  - GSM8K: grade-school math word problems with final-answer verification.
- **Methods**
  - `baseline`: direct input-output prompting.
  - `cot`: Chain-of-Thought prompting.
  - `self_consistency_cot`: multi-sample CoT voting for GSM8K.
  - `tot_bfs`: beam-search ToT for Game24 and GSM8K.
  - `tot_dfs`: depth-first ToT for Game24.
  - `tot_astar`: LLM-guided weighted A*-style ToT for Game24.
  - `tot_mcts`: value-based MCTS-style ToT for Game24.
- **Evaluation**
  - Per-sample JSON traces with raw model output, verification result, model calls, timing, token counts, and search metadata.
  - Aggregate accuracy and efficiency metrics saved under `results/`.

## Directory Structure

```text
.
├── config.py                  Default paths and experiment parameters
├── main.py                    CLI entry point
├── data/                      Local Game24 and GSM8K data files
├── datasets/                  Dataset loaders and validation
├── evaluation/                Game24/GSM8K evaluators and result saving
├── models/                    Model backend interface and Ollama backend
├── prompts/                   Prompt templates for all tasks and methods
├── solvers/                   Baseline, CoT, and ToT solvers
│   ├── game24_tot/            Game24 BFS/DFS/A*/MCTS ToT implementation
│   └── gsm8k_tot/             GSM8K ToT-BFS implementation
├── utils/                     Parsing and result I/O helpers
├── verifier/                  Exact answer verifiers
└── experiments.txt            Example experiment command log
```

## Environment

Install Python dependencies:

```powershell
pip install requests
```

Install [Ollama](https://ollama.com/) and pull a Qwen model:

```powershell
ollama pull qwen2.5:3b
```

The default model is configured in `config.py` as:

```text
qwen2.5:3b
```

You can override it with `--model`.

## Data

Default data paths are configured in `config.py`:

```text
Game24: data/game24.jsonl
GSM8K:  data/gsm8K.json
```

Game24 samples are JSONL records:

```json
{
  "id": "official-0001",
  "numbers": [1, 1, 1, 8],
  "solutions": ["(1+1+1)×8"],
  "solvable": true
}
```

GSM8K samples are JSON records or JSONL records:

```json
{
  "id": "gsm8k-test-1318",
  "question": "...",
  "answer": "... #### 14",
  "final_answer": "14"
}
```

The GSM8K loader excludes the first five records from evaluation because those records are used as fixed few-shot examples in `prompts/gsm8k_prompt.py`.

## Quick Start

Run commands from the repository root:

```powershell
cd D:\files\NJU\ToT\Tot_mcts
```

Game24:

```powershell
python main.py --dataset game24 --method baseline --limit 10
python main.py --dataset game24 --method cot --limit 10
python main.py --dataset game24 --method tot_bfs --limit 10
python main.py --dataset game24 --method tot_astar --limit 10
python main.py --dataset game24 --method tot_mcts --limit 10
```

GSM8K:

```powershell
python main.py --dataset gsm8k --method baseline --limit 10
python main.py --dataset gsm8k --method cot --limit 10
python main.py --dataset gsm8k --method self_consistency_cot --limit 10
python main.py --dataset gsm8k --method tot_bfs --limit 10
```

Run all methods supported by a dataset on the same sampled problems:

```powershell
python main.py --dataset game24 --method all --limit 10
python main.py --dataset gsm8k --method all --limit 10
```

Use `--no-shuffle` for deterministic first-N evaluation without dataset shuffling.

## Important Parameters

General:

```text
--dataset                         game24 or gsm8k
--method                          baseline, cot, self_consistency_cot,
                                  tot_bfs, tot_dfs, tot_astar, tot_mcts, all
--limit                           number of samples to evaluate
--seed                            dataset/model/search random seed
--model                           Ollama model name
--data-path                       override default dataset path
--results-dir                     output root for result JSON files
--ollama-url                      Ollama server URL
--temperature                     model generation temperature
--max-tokens                      default generation budget
--num-ctx                         Ollama context window
```

Prompting:

```text
--baseline-max-tokens             Game24 baseline generation budget
--cot-max-tokens                  Game24 CoT generation budget
--gsm8k-baseline-max-tokens       GSM8K baseline generation budget
--gsm8k-cot-max-tokens            GSM8K CoT generation budget
--self-consistency-samples        number of CoT samples for voting
--self-consistency-temperature    sampling temperature for CoT-SC
```

ToT search:

```text
--beam-width                      beam width for BFS-style ToT
--dfs-branch-limit                ranked children explored per DFS node
--astar-weight                    model heuristic weight for A*-style search
--mcts-iterations                 MCTS iteration budget per puzzle
--mcts-exploration                UCT exploration constant
--value-batch-size                states scored per model call
--value-max-tokens                value-scoring generation budget
--max-expanded-nodes              expansion cap per sample
```

GSM8K ToT:

```text
--gsm8k-tot-branch-factor         next-step candidates per state
--gsm8k-tot-max-depth             maximum reasoning depth
--gsm8k-tot-step-max-tokens       next-step generation budget
--gsm8k-tot-generation-temperature
                                  next-step generation temperature
```

## Algorithm Notes

- `tot_bfs` is beam BFS, not exhaustive BFS.
- `tot_astar` uses model scores as a weighted heuristic. The heuristic is not guaranteed to be admissible, so classical A* optimality guarantees do not apply.
- `tot_mcts` uses value estimates from the LLM instead of random rollout rewards.
- GSM8K currently supports baseline, CoT, Self-Consistency CoT, and ToT-BFS. MCTS and A* are implemented for Game24, where state transitions and exact verification are much easier to formalize.

## Results

Results are saved under:

```text
results/<dataset>/<method>/
```

Each result JSON contains:

- run configuration;
- per-sample question/input;
- raw model output;
- parsed prediction;
- verification result;
- duration, token counts, and model-call counts;
- search metadata such as expanded and generated nodes when applicable.

`results/` is ignored by Git because experiment traces can become large.

## Example Experiment Commands

See `experiments.txt` for the local command log used during experiments.

Typical GSM8K ToT-BFS sweep:

```powershell
python main.py --dataset gsm8k --method tot_bfs --limit 100 --beam-width 3 --gsm8k-tot-branch-factor 3 --gsm8k-tot-max-depth 6 --max-expanded-nodes 100 --value-max-tokens 256
python main.py --dataset gsm8k --method tot_bfs --limit 100 --beam-width 5 --gsm8k-tot-branch-factor 5 --gsm8k-tot-max-depth 8 --max-expanded-nodes 200 --value-max-tokens 256
```

## Verification

Before running expensive experiments, check that the CLI is available:

```powershell
python main.py --help
```

Then run a small smoke test while Ollama is running:

```powershell
python main.py --dataset game24 --method baseline --limit 1
python main.py --dataset gsm8k --method baseline --limit 1
```
