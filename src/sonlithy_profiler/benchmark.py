"""Benchmark orchestration for SonLithy Inference Profiler."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from sonlithy_profiler.config import BenchmarkConfig
from sonlithy_profiler.metrics import (
    Measurement,
    SummaryStatistics,
    calculate_generation_throughput,
    calculate_summary_statistics,
    calculate_total_request_latency,
)


@dataclass(frozen=True)
class InferenceRequest:
    """Backend-independent inference request."""

    prompt: str


@dataclass(frozen=True)
class InferenceResponse:
    """Response data used by the benchmark engine to calculate metrics."""

    generated_tokens: int | None = None
    generation_duration_seconds: Measurement = None
    time_to_first_token_seconds: Measurement = None


class InferenceBackend(Protocol):
    """Interface implemented by inference backends."""

    def execute(self, request: InferenceRequest) -> InferenceResponse:
        """Execute one inference request and return its metric data."""


@dataclass(frozen=True)
class IterationResult:
    """Outcome and measurements for one requested measured iteration."""

    iteration: int
    total_request_latency_seconds: float | None
    time_to_first_token_seconds: float | None
    generation_throughput_tokens_per_second: float | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the backend returned a valid measured response."""
        return self.error is None


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured iteration outcomes and their aggregate statistics."""

    config: BenchmarkConfig
    iterations: tuple[IterationResult, ...]
    warmup_failures: tuple[str, ...]
    summary: dict[str, SummaryStatistics]

    @property
    def successful_iterations(self) -> int:
        """Number of measured iterations with a valid response."""
        return sum(iteration.succeeded for iteration in self.iterations)

    @property
    def failed_iterations(self) -> int:
        """Number of measured iterations that failed."""
        return len(self.iterations) - self.successful_iterations


def _error_message(error: Exception) -> str:
    """Format an exception without turning it into a measurement."""
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _execute(
    backend: InferenceBackend, request: InferenceRequest
) -> InferenceResponse:
    response = backend.execute(request)
    if not isinstance(response, InferenceResponse):
        raise TypeError("backend must return an InferenceResponse")
    return response


def run_benchmark(
    config: BenchmarkConfig,
    backend: InferenceBackend,
    *,
    clock: Callable[[], float] = perf_counter,
) -> BenchmarkResult:
    """Run warm-ups followed by measured iterations using the given backend."""
    request = InferenceRequest(prompt=config.prompt)
    warmup_failures: list[str] = []
    for _ in range(config.warmup_iterations):
        try:
            _execute(backend, request)
        except Exception as error:
            warmup_failures.append(_error_message(error))

    measured: list[IterationResult] = []
    for iteration in range(1, config.iterations + 1):
        try:
            started_at = clock()
            response = _execute(backend, request)
            completed_at = clock()
            latency = calculate_total_request_latency(started_at, completed_at)
            throughput = calculate_generation_throughput(
                response.generated_tokens, response.generation_duration_seconds
            )
            ttft = response.time_to_first_token_seconds
            if ttft is not None:
                ttft = calculate_summary_statistics([ttft])["mean"]
            measured.append(
                IterationResult(
                    iteration=iteration,
                    total_request_latency_seconds=latency,
                    time_to_first_token_seconds=ttft,
                    generation_throughput_tokens_per_second=throughput,
                )
            )
        except Exception as error:
            measured.append(
                IterationResult(
                    iteration=iteration,
                    total_request_latency_seconds=None,
                    time_to_first_token_seconds=None,
                    generation_throughput_tokens_per_second=None,
                    error=_error_message(error),
                )
            )

    summary = {
        "total_request_latency_seconds": calculate_summary_statistics(
            result.total_request_latency_seconds
            for result in measured
            if result.succeeded
        ),
        "time_to_first_token_seconds": calculate_summary_statistics(
            result.time_to_first_token_seconds
            for result in measured
            if result.succeeded
        ),
        "generation_throughput_tokens_per_second": calculate_summary_statistics(
            result.generation_throughput_tokens_per_second
            for result in measured
            if result.succeeded
        ),
    }
    return BenchmarkResult(
        config=config,
        iterations=tuple(measured),
        warmup_failures=tuple(warmup_failures),
        summary=summary,
    )
