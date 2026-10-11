import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from sonlithy_profiler.backends.ollama import OllamaBackend
from sonlithy_profiler.benchmark import InferenceRequest, run_benchmark
from sonlithy_profiler.config import BenchmarkConfig


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        pass


def backend_response(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> list[tuple[Request, float]]:
    calls: list[tuple[Request, float]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        calls.append((request, timeout))
        return FakeHTTPResponse(body)

    monkeypatch.setattr(
        "sonlithy_profiler.backends.ollama.urlopen", fake_urlopen
    )
    return calls


def test_successful_response_uses_generation_metadata_and_nanoseconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = backend_response(
        monkeypatch,
        json.dumps(
            {
                "response": "generated text",
                "done": True,
                "prompt_eval_count": 100,
                "eval_count": 25,
                "eval_duration": 2_500_000_000,
            }
        ).encode(),
    )
    backend = OllamaBackend("http://ollama.local:11434/", "model-name", timeout=3)

    response = backend.execute(InferenceRequest("test prompt"))

    assert response.generated_tokens == 25
    assert response.generation_duration_seconds == 2.5
    assert response.time_to_first_token_seconds is None
    sent_request, timeout = calls[0]
    assert sent_request.full_url == "http://ollama.local:11434/api/generate"
    assert json.loads(sent_request.data) == {
        "model": "model-name",
        "prompt": "test prompt",
        "stream": False,
    }
    assert timeout == 3.0


def test_absent_optional_metadata_returns_unavailable_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_response(monkeypatch, b'{"response": "text", "done": true}')
    response = OllamaBackend("http://localhost:11434", "model").execute(
        InferenceRequest("prompt")
    )

    assert response.generated_tokens is None
    assert response.generation_duration_seconds is None
    assert response.time_to_first_token_seconds is None


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b'{"response": 1, "done": true}',
        b'{"response": "text", "done": false}',
    ],
)
def test_incomplete_ollama_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    backend_response(monkeypatch, body)
    with pytest.raises(ValueError, match="Ollama response"):
        OllamaBackend("http://localhost:11434", "model").execute(
            InferenceRequest("prompt")
        )


@pytest.mark.parametrize(
    "base_url",
    ["http://:11434", "http://localhost:not-a-port", "http://localhost:99999"],
)
def test_invalid_host_urls_are_rejected(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        OllamaBackend(base_url, "model")


def test_invalid_json_is_reported_as_a_failed_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_response(monkeypatch, b"{not-json")
    result = run_benchmark(
        BenchmarkConfig("prompt"),
        OllamaBackend("http://localhost:11434", "model"),
        clock=iter((1.0,)).__next__,
    )

    assert result.failed_iterations == 1
    assert result.iterations[0].error == "ValueError: Ollama returned invalid JSON"


def test_http_error_is_reported_as_a_failed_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_http_error(request: Request, timeout: float) -> FakeHTTPResponse:
        raise HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=FakeHTTPResponse(b'{"error":"model not found"}'),
        )

    monkeypatch.setattr(
        "sonlithy_profiler.backends.ollama.urlopen", raise_http_error
    )
    result = run_benchmark(
        BenchmarkConfig("prompt"),
        OllamaBackend("http://localhost:11434", "missing-model"),
        clock=iter((1.0,)).__next__,
    )

    assert result.failed_iterations == 1
    assert result.iterations[0].error == (
        'RuntimeError: Ollama returned HTTP 404: {"error":"model not found"}'
    )


def test_timeout_is_reported_as_a_failed_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(request: Request, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError("request timed out")

    monkeypatch.setattr(
        "sonlithy_profiler.backends.ollama.urlopen", raise_timeout
    )
    result = run_benchmark(
        BenchmarkConfig("prompt"),
        OllamaBackend("http://localhost:11434", "model"),
        clock=iter((1.0,)).__next__,
    )

    assert result.failed_iterations == 1
    assert result.iterations[0].error == "TimeoutError: request timed out"


def test_ollama_error_response_is_reported_as_a_failed_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_response(monkeypatch, b'{"error":"model not found"}')
    result = run_benchmark(
        BenchmarkConfig("prompt"),
        OllamaBackend("http://localhost:11434", "missing-model"),
        clock=iter((1.0,)).__next__,
    )

    assert result.failed_iterations == 1
    assert result.iterations[0].error == "RuntimeError: Ollama error: model not found"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eval_count", -1),
        ("eval_count", 1.5),
        ("eval_count", True),
        ("eval_duration", -1),
        ("eval_duration", 1.5),
        ("eval_duration", True),
    ],
)
def test_invalid_numeric_metadata_is_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    backend_response(
        monkeypatch,
        json.dumps({"response": "text", "done": True, field: value}).encode(),
    )
    with pytest.raises(ValueError, match=f"Ollama {field} must be a non-negative integer"):
        OllamaBackend("http://localhost:11434", "model").execute(
            InferenceRequest("prompt")
        )


def test_zero_generation_duration_is_not_divided_by_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_response(
        monkeypatch,
        b'{"response":"text","done":true,"eval_count":1,"eval_duration":0}',
    )
    result = run_benchmark(
        BenchmarkConfig("prompt"),
        OllamaBackend("http://localhost:11434", "model"),
        clock=iter((1.0, 2.0)).__next__,
    )

    assert result.failed_iterations == 1
    assert result.iterations[0].generation_throughput_tokens_per_second is None
    assert result.iterations[0].error == (
        "ValueError: generation_duration_seconds must be greater than zero"
    )
