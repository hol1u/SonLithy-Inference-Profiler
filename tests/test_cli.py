import json
from pathlib import Path

import pytest

from sonlithy_profiler import __version__
from sonlithy_profiler.cli import PROJECT_NAME, main


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return self.body


def test_cli_displays_project_name_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == f"{PROJECT_NAME} {__version__}"


def test_cli_version_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--version"])

    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == f"{PROJECT_NAME} {__version__}"


def test_cli_help_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])

    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: sonlithy-profiler" in help_text
    assert f"{PROJECT_NAME} {__version__}" in help_text


def test_cli_runs_benchmark_and_writes_json_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from urllib.request import Request

    responses = [
        b'{"response":"warmup","done":true,"eval_count":100,"eval_duration":1000000000}',
        b'{"response":"generated","done":true,"eval_count":10,"eval_duration":2000000000}',
        b'{"error":"model unavailable"}',
    ]
    requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        requests.append(request)
        assert timeout == 12.0
        return FakeHTTPResponse(responses.pop(0))

    monkeypatch.setattr(
        "sonlithy_profiler.backends.ollama.urlopen", fake_urlopen
    )
    report_path = tmp_path / "benchmark.json"

    exit_code = main(
        [
            "benchmark",
            "--model",
            "llama-test",
            "--prompt",
            "test prompt",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--host",
            "http://ollama.test:11434",
            "--timeout",
            "12",
            "--output",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert len(requests) == 3
    assert json.loads(requests[0].data) == {
        "model": "llama-test",
        "prompt": "test prompt",
        "stream": False,
    }
    assert "Model: llama-test" in captured.out
    assert "Iterations: 1 successful, 1 failed" in captured.out
    assert "Throughput: 5.000 tokens/s" in captured.out
    assert "TTFT: N/A" in captured.out
    assert "Iteration 2 failed: RuntimeError: Ollama error: model unavailable" in (
        captured.err
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["backend"]["host"] == "http://ollama.test:11434"
    assert report["configuration"] == {
        "prompt": "test prompt",
        "iterations": 2,
        "warmup_iterations": 1,
    }
    assert report["counts"] == {
        "requested_iterations": 2,
        "successful_iterations": 1,
        "failed_iterations": 1,
        "warmup_failures": 0,
    }
    assert report["summary"]["time_to_first_token_seconds"]["mean"] is None
    assert report["iterations"][1]["error"] == (
        "RuntimeError: Ollama error: model unavailable"
    )


def test_cli_rejects_invalid_iteration_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as result:
        main(
            [
                "benchmark",
                "--model",
                "llama-test",
                "--prompt",
                "prompt",
                "--iterations",
                "0",
            ]
        )

    assert result.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_cli_rejects_non_finite_timeout(
    capsys: pytest.CaptureFixture[str], timeout: str
) -> None:
    with pytest.raises(SystemExit) as result:
        main(
            [
                "benchmark",
                "--model",
                "llama-test",
                "--prompt",
                "prompt",
                f"--timeout={timeout}",
            ]
        )

    assert result.value.code == 2
    assert "must be a positive number" in capsys.readouterr().err


def test_cli_rejects_empty_output_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as result:
        main(
            [
                "benchmark",
                "--model",
                "llama-test",
                "--prompt",
                "prompt",
                "--output",
                "",
            ]
        )

    assert result.value.code == 2
    assert "must not be empty" in capsys.readouterr().err


def test_cli_returns_nonzero_when_report_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    response = b'{"response":"generated","done":true,"eval_count":1,"eval_duration":1000000000}'

    def fake_urlopen(request: object, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(response)

    monkeypatch.setattr(
        "sonlithy_profiler.backends.ollama.urlopen", fake_urlopen
    )
    report_path = tmp_path / "missing" / "report.json"

    exit_code = main(
        [
            "benchmark",
            "--model",
            "llama-test",
            "--prompt",
            "prompt",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 1
    assert "Error writing JSON report:" in capsys.readouterr().err
