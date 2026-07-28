"""Evaluation loop for Game24 solvers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from datasets.game24 import Game24Sample
from solvers.base_solver import BaseSolver
from utils.result_io import save_json
from verifier.game24 import (
    VerificationResult,
    verify_24_expression,
)


def evaluate_game24(
    *,
    samples: list[Game24Sample],
    solver: BaseSolver,
    model_name: str,
    seed: int,
    results_dir: str | Path,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one solver on Game24 samples and save result JSON."""
    sample_results: list[dict[str, Any]] = []

    correct_count = 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_duration_seconds = 0.0

    # Baseline and CoT leave search metrics at zero.
    total_model_calls = 0
    total_expanded_nodes = 0
    total_generated_nodes = 0

    total_samples = len(samples)

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        # Give each sample a distinct but reproducible generation seed.
        sample_seed = seed + index - 1

        try:
            solver_result = solver.solve(
                sample,
                seed=sample_seed,
            )

            expression = solver_result.expression

            if expression is None:
                verification = VerificationResult(
                    ok=False,
                    reason=(
                        "could not obtain a final expression "
                        "from solver"
                    ),
                )
            else:
                verification = verify_24_expression(
                    expression,
                    sample.numbers,
                )

            error_message = None

        except Exception as exc:
            # Keep long experiment runs going when one sample fails.
            expression = None

            verification = VerificationResult(
                ok=False,
                reason=f"solver error: {exc}",
            )

            solver_result = None
            error_message = str(exc)

        if verification.ok:
            correct_count += 1

        if solver_result is not None:
            prompt_tokens = solver_result.prompt_tokens
            completion_tokens = (
                solver_result.completion_tokens
            )
            duration_seconds = (
                solver_result.duration_seconds
            )

            if prompt_tokens is not None:
                total_prompt_tokens += prompt_tokens

            if completion_tokens is not None:
                total_completion_tokens += (
                    completion_tokens
                )

            if duration_seconds is not None:
                total_duration_seconds += (
                    duration_seconds
                )

            solver_metadata = solver_result.metadata

        else:
            prompt_tokens = None
            completion_tokens = None
            duration_seconds = None
            solver_metadata = {}

        # Extract search metrics when the solver provides them.
        search_metadata = solver_metadata.get(
            "search",
            {},
        )

        expanded_nodes = _safe_int(
            search_metadata.get("expanded_nodes")
        )

        generated_nodes = _safe_int(
            search_metadata.get("generated_nodes")
        )

        model_calls = (
            solver_result.model_calls
            if solver_result is not None
            else 0
        )

        total_expanded_nodes += expanded_nodes
        total_generated_nodes += generated_nodes
        total_model_calls += model_calls

        print(
            f"[{index}/{total_samples}] "
            f"id={sample.id} "
            f"input={sample.input_text}"
        )

        print(
            "Expression: "
            f"{expression if expression is not None else '<not found>'}"
        )

        status = (
            "PASS"
            if verification.ok
            else "FAIL"
        )

        print(
            f"Verification: {status} "
            f"({verification.reason})"
        )

        if search_metadata:
            print(
                "Search: "
                f"expanded={expanded_nodes}, "
                f"generated={generated_nodes}, "
                f"model_calls={model_calls}"
            )

        print("-" * 72)

        sample_result = {
            "index": index,
            "id": sample.id,
            "numbers": list(sample.numbers),
            "input_text": sample.input_text,
            "sample_seed": sample_seed,
            "expression": expression,
            "raw_output": _compact_raw_output(
                solver_metadata,
                solver_result.raw_response,
            )
            if solver_result is not None
            else "",
            "llm_calls": model_calls,
            "is_correct": verification.ok,
            "verification": {
                "ok": verification.ok,
                "value": (
                    str(verification.value)
                    if verification.value is not None
                    else None
                ),
                "reason": verification.reason,
            },
            "dataset_metadata": sample.metadata,
        }

        if duration_seconds is not None:
            sample_result["duration_seconds"] = (
                duration_seconds
            )

        if expanded_nodes or generated_nodes:
            sample_result["search"] = {
                "expanded_nodes": expanded_nodes,
                "generated_nodes": generated_nodes,
            }

        if error_message is not None:
            sample_result["error"] = error_message

        sample_results.append(sample_result)

    accuracy = (
        correct_count / total_samples
        if total_samples > 0
        else 0.0
    )

    average_duration_seconds = (
        total_duration_seconds / total_samples
        if total_samples > 0
        else 0.0
    )

    average_expanded_nodes = (
        total_expanded_nodes / total_samples
        if total_samples > 0
        else 0.0
    )

    average_generated_nodes = (
        total_generated_nodes / total_samples
        if total_samples > 0
        else 0.0
    )

    average_model_calls = (
        total_model_calls / total_samples
        if total_samples > 0
        else 0.0
    )

    print(
        f"Accuracy: {correct_count}/{total_samples} "
        f"= {accuracy:.2%}"
    )

    if solver.name.startswith("tot_"):
        print(
            "Search summary: "
            f"expanded={total_expanded_nodes}, "
            f"generated={total_generated_nodes}, "
            f"model_calls={total_model_calls}"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # Keep model names safe for filenames.
    safe_model_name = (
        model_name
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_filename = (
        f"{solver.name}_"
        f"{safe_model_name}_"
        f"seed{seed}_"
        f"limit{total_samples}_"
        f"{timestamp}.json"
    )

    output_path = _build_output_path(
        results_dir=results_dir,
        solver_name=solver.name,
        output_filename=output_filename,
    )

    result_data = {
        "summary": {
            "method": solver.name,
            "model": model_name,
            "dataset": "game24",
            "seed": seed,
            "total": total_samples,
            "correct": correct_count,

            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,

            "total_prompt_tokens": (
                total_prompt_tokens
            ),
            "total_completion_tokens": (
                total_completion_tokens
            ),
            "total_duration_seconds": (
                total_duration_seconds
            ),
            "average_duration_seconds": (
                average_duration_seconds
            ),

            "total_model_calls": total_model_calls,
            "average_model_calls": (
                average_model_calls
            ),
            "total_expanded_nodes": (
                total_expanded_nodes
            ),
            "average_expanded_nodes": (
                average_expanded_nodes
            ),
            "total_generated_nodes": (
                total_generated_nodes
            ),
            "average_generated_nodes": (
                average_generated_nodes
            ),
        },
        "config": run_config or {},
        "timestamp": timestamp,
        "samples": sample_results,
    }

    saved_path = save_json(
        result_data,
        output_path,
    )

    print(f"Results saved to: {saved_path}")

    return result_data


def _build_output_path(
    *,
    results_dir: str | Path,
    solver_name: str,
    output_filename: str,
) -> Path:
    """Build the result path, grouping ToT outputs by strategy."""
    base_path = Path(results_dir) / "game24"

    if solver_name.startswith("tot_"):
        strategy = solver_name.removeprefix(
            "tot_"
        )

        return (
            base_path
            / "tot"
            / strategy
            / output_filename
        )

    return (
        base_path
        / solver_name
        / output_filename
    )


def _safe_int(
    value: Any,
) -> int:
    """Convert optional metadata counters to integers."""
    if isinstance(value, bool):
        return 0

    if isinstance(value, int):
        return value

    return 0


def _compact_raw_output(
    solver_metadata: dict[str, Any],
    raw_response: str,
) -> str:
    """
    Build a compact per-sample raw output string for result JSON.

    Baseline and CoT do not need their full prompt/metadata here.
    ToT keeps the model's raw rating output plus selected states.
    """
    search_metadata = solver_metadata.get(
        "search",
        {},
    )

    value_metadata = solver_metadata.get(
        "value_evaluator",
        {},
    )

    trace = search_metadata.get("trace")
    logs = value_metadata.get("logs")

    if isinstance(trace, list) and isinstance(logs, list):
        compact_trace = _compact_tot_trace(
            trace,
            logs,
        )

        return _json_dumps(compact_trace)

    return raw_response


def _compact_tot_trace(
    trace: list[Any],
    logs: list[Any],
) -> list[dict[str, Any]]:
    compact_trace: list[dict[str, Any]] = []
    log_index = 0

    for event in trace:
        if not isinstance(event, dict):
            continue

        compact_event: dict[str, Any] = {}

        if "depth" in event:
            compact_event["depth"] = event["depth"]

        if "candidate_count" in event:
            compact_event["candidate_count"] = (
                event["candidate_count"]
            )

        if "event" in event:
            compact_event["event"] = event["event"]

        raw_value_output = _next_raw_value_output(
            logs,
            log_index,
        )

        if raw_value_output is not None:
            compact_event["raw_value_output"] = (
                raw_value_output
            )
            log_index += 1

        selected = _compact_selected_states(event)

        if selected:
            compact_event["selected"] = selected

        if compact_event:
            compact_trace.append(compact_event)

    return compact_trace


def _next_raw_value_output(
    logs: list[Any],
    log_index: int,
) -> str | None:
    if log_index >= len(logs):
        return None

    log = logs[log_index]

    if not isinstance(log, dict):
        return None

    raw_response = log.get("raw_response")

    return raw_response if isinstance(raw_response, str) else None


def _compact_selected_states(
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = event.get("selected")

    if selected is None:
        selected = event.get("children")

    if not isinstance(selected, list):
        return []

    compact_states: list[dict[str, Any]] = []

    for item in selected:
        if not isinstance(item, dict):
            continue

        state = item.get("state")

        if not isinstance(state, dict):
            continue

        compact_states.append(
            {
                "numbers": " ".join(
                    str(number)
                    for number in state.get("numbers", [])
                ),
                "expressions": _format_terms(
                    state.get("terms", [])
                ),
                "score": item.get("score"),
            }
        )

    return compact_states


def _format_terms(
    terms: Any,
) -> str:
    if not isinstance(terms, list):
        return ""

    expressions: list[str] = []

    for term in terms:
        if not isinstance(term, dict):
            continue

        expression = term.get("expression")

        if isinstance(expression, str):
            expressions.append(expression)

    return ", ".join(expressions)


def _json_dumps(
    value: Any,
) -> str:
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
    )
