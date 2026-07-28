"""Ollama model backend."""

from __future__ import annotations

import time
from typing import Any

import requests

from models.base_model import BaseModel, GenerationResult


class OllamaModel(BaseModel):
    """Ollama implementation of the model interface."""

    def __init__(
        self,
        *,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        default_temperature: float = 0.0,
        default_max_tokens: int = 512,
        default_context_length: int = 2048,
        keep_alive: str = "30m",
        timeout: int = 300,
    ) -> None:
        # Normalize the base URL before appending API paths.
        self.base_url = base_url.rstrip("/")

        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_context_length = default_context_length
        self.keep_alive = keep_alive
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        context_length: int | None = None,
    ) -> GenerationResult:
        selected_temperature = (
            self.default_temperature
            if temperature is None
            else temperature
        )

        selected_max_tokens = (
            self.default_max_tokens
            if max_tokens is None
            else max_tokens
        )

        selected_context_length = (
            self.default_context_length
            if context_length is None
            else context_length
        )

        options: dict[str, Any] = {
            "temperature": selected_temperature,

            # Ollama uses num_predict for the generation limit.
            "num_predict": selected_max_tokens,
            "num_ctx": selected_context_length,
        }

        if seed is not None:
            options["seed"] = seed

        payload = {
            "model": self.model_name,
            "prompt": prompt,

            # Keep response handling simple and deterministic.
            "stream": False,

            "keep_alive": self.keep_alive,
            "options": options,
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running at "
                f"{self.base_url}."
            ) from exc

        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        except requests.HTTPError as exc:
            response_text = (
                response.text
                if response is not None
                else ""
            )

            model_hint = ""
            if response is not None and response.status_code == 404:
                model_hint = (
                    " If the model has not been downloaded, run: "
                    f"ollama pull {self.model_name}"
                )

            raise RuntimeError(
                "Ollama returned an HTTP error: "
                f"{response_text}{model_hint}"
            ) from exc

        duration_seconds = time.perf_counter() - start_time

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from exc

        generated_text = data.get("response")

        if not isinstance(generated_text, str):
            raise RuntimeError(
                "Ollama response does not contain a valid "
                "'response' field."
            )

        ollama_duration = data.get("total_duration")

        if isinstance(ollama_duration, int):
            reported_duration_seconds = (
                ollama_duration / 1_000_000_000
            )
        else:
            reported_duration_seconds = duration_seconds

        return GenerationResult(
            text=generated_text.strip(),
            model=str(data.get("model", self.model_name)),
            prompt_tokens=_optional_int(
                data.get("prompt_eval_count")
            ),
            completion_tokens=_optional_int(
                data.get("eval_count")
            ),
            duration_seconds=reported_duration_seconds,
            metadata={
                "load_duration": data.get("load_duration"),
                "prompt_eval_duration": data.get(
                    "prompt_eval_duration"
                ),
                "eval_duration": data.get("eval_duration"),
                "done_reason": data.get("done_reason"),
            },
        )


def _optional_int(value: Any) -> int | None:
    """Return an integer value or None."""
    return value if isinstance(value, int) else None
