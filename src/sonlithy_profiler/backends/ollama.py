"""Ollama HTTP backend integration."""

import json
from math import isfinite
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from sonlithy_profiler.benchmark import InferenceRequest, InferenceResponse

_NANOSECONDS_PER_SECOND = 1_000_000_000


class OllamaBackend:
    """Run non-streaming inference requests against an Ollama server."""

    def __init__(
        self, base_url: str, model: str, timeout: float = 60.0
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty HTTP(S) URL")
        parsed_url = urlsplit(base_url)
        try:
            hostname = parsed_url.hostname
            parsed_url.port
        except ValueError as error:
            raise ValueError("base_url must be a valid HTTP(S) URL") from error
        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.netloc
            or not hostname
        ):
            raise ValueError("base_url must be a valid HTTP(S) URL")
        if parsed_url.query or parsed_url.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")

        self._endpoint = f"{base_url.rstrip('/')}/api/generate"
        self._model = model
        self._timeout = float(timeout)

    def execute(self, request: InferenceRequest) -> InferenceResponse:
        """Send one prompt and normalize Ollama's generation metadata."""
        payload = json.dumps(
            {"model": self._model, "prompt": request.prompt, "stream": False}
        ).encode("utf-8")
        http_request = Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self._timeout) as response:
                response_body = response.read()
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace").strip()
            details = f": {error_body}" if error_body else ""
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}{details}"
            ) from error

        try:
            result: Any = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Ollama returned invalid JSON") from error
        if not isinstance(result, dict):
            raise ValueError("Ollama response must be a JSON object")

        ollama_error = result.get("error")
        if ollama_error is not None:
            if not isinstance(ollama_error, str):
                raise ValueError("Ollama error field must be a string")
            if ollama_error:
                raise RuntimeError(f"Ollama error: {ollama_error}")

        if not isinstance(result.get("response"), str):
            raise ValueError("Ollama response must include a string response field")
        if result.get("done") is not True:
            raise ValueError("Ollama response must indicate done=true")

        generated_tokens = self._optional_non_negative_integer(
            result, "eval_count"
        )
        generation_duration_ns = self._optional_non_negative_integer(
            result, "eval_duration"
        )
        generation_duration_seconds = None
        if generation_duration_ns is not None:
            try:
                generation_duration_seconds = (
                    generation_duration_ns / _NANOSECONDS_PER_SECOND
                )
            except OverflowError as error:
                raise ValueError("Ollama eval_duration is out of range") from error
            if not isfinite(generation_duration_seconds):
                raise ValueError("Ollama eval_duration must be finite")

        return InferenceResponse(
            generated_tokens=generated_tokens,
            generation_duration_seconds=generation_duration_seconds,
            time_to_first_token_seconds=None,
        )

    @staticmethod
    def _optional_non_negative_integer(
        result: dict[str, Any], field: str
    ) -> int | None:
        """Validate an optional integer metadata field returned by Ollama."""
        value = result.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Ollama {field} must be a non-negative integer")
        return value
