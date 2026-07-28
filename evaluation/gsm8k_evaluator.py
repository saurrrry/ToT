"""Evaluation loop for GSM8K solvers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from datasets.gsm8k import GSM8KSample
from solvers.base_solver import BaseSolver
from utils.result_io import save_json
from verifier.gsm8k import verify_gsm8k_answer


def evaluate_gsm8k(
    *,
    samples: list[GSM8KSample],
    solver: BaseSolver,
    model_name: str,
    seed: int,
    results_dir: str | Path,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one solver on GSM8K samples and save result JSON."""
    sample_results: list[dict[str, Any]] = []

    correct_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_duration_seconds = 0.0
    total_model_calls = 0
    total_expanded_nodes = 0
    total_generated_nodes = 0
    total_samples = len(samples)

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        sample_seed = seed + index - 1

        try:
            solver_result = solver.solve(
                sample,
                seed=sample_seed,
            )
            predicted_answer = solver_result.expression

            verification = verify_gsm8k_answer(
                predicted_answer,
                sample.final_answer,
            )
            error_message = None

        except Exception as exc:
            solver_result = None
            predicted_answer = None
            verification = verify_gsm8k_answer(
                None,
                sample.final_answer,
            )
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
            model_calls = solver_result.model_calls
            metadata = solver_result.metadata
            raw_output = solver_result.raw_response

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

        else:
            prompt_tokens = None
            completion_tokens = None
            duration_seconds = None
            model_calls = 0
            metadata = {}
            raw_output = ""

        total_model_calls += model_calls

        search_metadata = metadata.get(
            "search",
            {},
        )
        expanded_nodes = _safe_int(
            search_metadata.get("expanded_nodes")
        )
        generated_nodes = _safe_int(
            search_metadata.get("generated_nodes")
        )

        total_expanded_nodes += expanded_nodes
        total_generated_nodes += generated_nodes

        print(
            f"[{index}/{total_samples}] "
            f"id={sample.id}"
        )
        print(
            "Prediction: "
            f"{predicted_answer if predicted_answer is not None else '<not found>'}"
        )
        print(f"Gold: {sample.final_answer}")
        print(
            "Verification: "
            f"{'PASS' if verification.ok else 'FAIL'} "
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
            "question": sample.question,
            "gold_answer": sample.final_answer,
            "predicted_answer": predicted_answer,
            "normalized_prediction": verification.predicted,
            "normalized_gold": verification.expected,
            "raw_output": raw_output,
            "llm_calls": model_calls,
            "is_correct": verification.ok,
            "verification": {
                "ok": verification.ok,
                "predicted": verification.predicted,
                "expected": verification.expected,
                "reason": verification.reason,
            },
            "dataset_metadata": sample.metadata,
            "solver_metadata": metadata,
            "sample_seed": sample_seed,
        }

        if duration_seconds is not None:
            sample_result["duration_seconds"] = (
                duration_seconds
            )

        if prompt_tokens is not None:
            sample_result["prompt_tokens"] = (
                prompt_tokens
            )

        if completion_tokens is not None:
            sample_result["completion_tokens"] = (
                completion_tokens
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

    average_model_calls = (
        total_model_calls / total_samples
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

    print(
        f"Accuracy: {correct_count}/{total_samples} "
        f"= {accuracy:.2%}"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

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

    output_path = (
        Path(results_dir)
        / "gsm8k"
        / solver.name
        / output_filename
    )

    result_data = {
        "summary": {
            "method": solver.name,
            "model": model_name,
            "dataset": "gsm8k",
            "seed": seed,
            "total": total_samples,
            "correct": correct_count,
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,
            "total_prompt_tokens": total_prompt_tokens,
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


def _safe_int(
    value: Any,
) -> int:
    """Convert optional metadata counters to integers."""
    if isinstance(value, bool):
        return 0

    if isinstance(value, int):
        return value

    return 0
