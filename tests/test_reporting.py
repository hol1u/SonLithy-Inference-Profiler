import json
from pathlib import Path

from sonlithy_profiler.benchmark import (
    InferenceRequest,
    InferenceResponse,
    run_benchmark,
)
from sonlithy_profiler.config import BenchmarkConfig
from sonlithy_profiler.reporting import json_report


def test_json_report_module_is_importable() -> None:
    assert json_report.__doc__


def test_json_report_serializes_unavailable_metrics_and_iteration_errors(
    tmp_path: Path,
) -> None:
    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            if self.calls == 1:
                return InferenceResponse()
            raise RuntimeError("backend unavailable")

    result = run_benchmark(
        BenchmarkConfig("prompt", iterations=2),
        Backend(),
        clock=iter((1.0, 2.0, 3.0)).__next__,
    )
    destination = tmp_path / "report.json"

    written_path = json_report.write_json_report(
        result,
        destination,
        model="model-name",
        host="http://localhost:11434",
        timeout=60.0,
    )

    assert written_path == destination
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["iterations"][0]["time_to_first_token_seconds"] is None
    assert (
        report["iterations"][0]["generation_throughput_tokens_per_second"] is None
    )
    assert report["iterations"][1]["total_request_latency_seconds"] is None
    assert report["iterations"][1]["error"] == (
        "RuntimeError: backend unavailable"
    )
    assert report["summary"]["total_request_latency_seconds"]["mean"] == 1.0
