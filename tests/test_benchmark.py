from dataclasses import dataclass, field

from sonlithy_profiler.benchmark import (
    InferenceRequest,
    InferenceResponse,
    run_benchmark,
)
from sonlithy_profiler.config import BenchmarkConfig


@dataclass
class FakeBackend:
    responses: list[InferenceResponse | Exception]
    requests: list[InferenceRequest] = field(default_factory=list)

    def execute(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_warmup_iterations_are_excluded_from_results_and_summary() -> None:
    backend = FakeBackend(
        [
            InferenceResponse(10_000, 1.0, 0.001),
            InferenceResponse(4, 2.0, 0.5),
            InferenceResponse(8, 2.0, 1.0),
        ]
    )
    result = run_benchmark(
        BenchmarkConfig("prompt", iterations=2, warmup_iterations=1),
        backend,
        clock=iter((1.0, 3.0, 10.0, 14.0)).__next__,
    )

    assert len(backend.requests) == 3
    assert len(result.iterations) == 2
    assert result.summary["total_request_latency_seconds"]["mean"] == 3.0
    assert result.summary["generation_throughput_tokens_per_second"]["mean"] == 3.0
    assert result.summary["time_to_first_token_seconds"]["mean"] == 0.75


def test_successful_run_records_each_iteration_and_uses_metric_summaries() -> None:
    backend = FakeBackend(
        [InferenceResponse(10, 2.0, 0.25), InferenceResponse(20, 4.0, 0.5)]
    )
    result = run_benchmark(
        BenchmarkConfig("prompt", iterations=2),
        backend,
        clock=iter((1.0, 2.0, 3.0, 5.0)).__next__,
    )

    assert result.successful_iterations == 2
    assert result.failed_iterations == 0
    assert [item.total_request_latency_seconds for item in result.iterations] == [
        1.0,
        2.0,
    ]
    assert result.summary["total_request_latency_seconds"] == {
        "mean": 1.5,
        "median": 1.5,
        "minimum": 1.0,
        "maximum": 2.0,
    }
    assert result.summary["generation_throughput_tokens_per_second"] == {
        "mean": 5.0,
        "median": 5.0,
        "minimum": 5.0,
        "maximum": 5.0,
    }


def test_partial_failures_are_recorded_without_zero_measurements() -> None:
    backend = FakeBackend(
        [
            InferenceResponse(4, 2.0),
            RuntimeError("backend unavailable"),
            InferenceResponse(6, 3.0),
        ]
    )
    result = run_benchmark(
        BenchmarkConfig("prompt", iterations=3),
        backend,
        clock=iter((1.0, 2.0, 3.0, 4.0, 5.0)).__next__,
    )

    assert result.successful_iterations == 2
    assert result.failed_iterations == 1
    failed = result.iterations[1]
    assert failed.error == "RuntimeError: backend unavailable"
    assert failed.total_request_latency_seconds is None
    assert failed.generation_throughput_tokens_per_second is None
    assert result.summary["total_request_latency_seconds"]["mean"] == 1.0


def test_all_iterations_failing_returns_empty_summaries() -> None:
    backend = FakeBackend([RuntimeError("first"), RuntimeError("second")])
    result = run_benchmark(
        BenchmarkConfig("prompt", iterations=2),
        backend,
        clock=iter((1.0, 2.0)).__next__,
    )

    assert result.successful_iterations == 0
    assert result.failed_iterations == 2
    assert all(not item.succeeded for item in result.iterations)
    assert all(
        statistics
        == {"mean": None, "median": None, "minimum": None, "maximum": None}
        for statistics in result.summary.values()
    )


def test_warmup_failures_are_retained_separately() -> None:
    backend = FakeBackend([RuntimeError("warmup failed"), InferenceResponse(2, 1.0)])
    result = run_benchmark(
        BenchmarkConfig("prompt", warmup_iterations=1),
        backend,
        clock=iter((2.0, 3.0)).__next__,
    )

    assert result.warmup_failures == ("RuntimeError: warmup failed",)
    assert len(result.iterations) == 1
    assert result.iterations[0].succeeded


def test_invalid_backend_measurements_are_recorded_as_failures() -> None:
    backend = FakeBackend([InferenceResponse(2, 0)])
    result = run_benchmark(
        BenchmarkConfig("prompt"),
        backend,
        clock=iter((1.0, 2.0)).__next__,
    )

    assert result.failed_iterations == 1
    assert "generation_duration_seconds" in result.iterations[0].error


def test_backend_must_return_typed_response() -> None:
    class InvalidBackend:
        def execute(self, request: InferenceRequest) -> InferenceResponse:
            return {"generated_tokens": 1}  # type: ignore[return-value]

    result = run_benchmark(
        BenchmarkConfig("prompt"), InvalidBackend(), clock=iter((1.0,)).__next__
    )

    assert result.iterations[0].error == (
        "TypeError: backend must return an InferenceResponse"
    )


def test_configuration_is_preserved_in_result() -> None:
    config = BenchmarkConfig("prompt", iterations=1)
    result = run_benchmark(
        config,
        FakeBackend([InferenceResponse()]),
        clock=iter((1.0, 2.0)).__next__,
    )

    assert result.config is config
