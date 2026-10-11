"""Serialize benchmark results as machine-readable JSON reports."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sonlithy_profiler import __version__
from sonlithy_profiler.benchmark import BenchmarkResult


def benchmark_result_to_dict(
    result: BenchmarkResult,
    *,
    model: str,
    host: str,
    timeout: float,
) -> dict[str, Any]:
    """Convert a benchmark result into a JSON-compatible report structure."""
    return {
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": {
            "name": "ollama",
            "host": host,
            "model": model,
            "timeout_seconds": timeout,
        },
        "configuration": {
            "prompt": result.config.prompt,
            "iterations": result.config.iterations,
            "warmup_iterations": result.config.warmup_iterations,
        },
        "counts": {
            "requested_iterations": result.config.iterations,
            "successful_iterations": result.successful_iterations,
            "failed_iterations": result.failed_iterations,
            "warmup_failures": len(result.warmup_failures),
        },
        "summary": result.summary,
        "iterations": [
            {
                "iteration": iteration.iteration,
                "succeeded": iteration.succeeded,
                "total_request_latency_seconds": (
                    iteration.total_request_latency_seconds
                ),
                "time_to_first_token_seconds": (
                    iteration.time_to_first_token_seconds
                ),
                "generation_throughput_tokens_per_second": (
                    iteration.generation_throughput_tokens_per_second
                ),
                "error": iteration.error,
            }
            for iteration in result.iterations
        ],
        "warmup_errors": list(result.warmup_failures),
    }


def write_json_report(
    result: BenchmarkResult,
    path: str | Path,
    *,
    model: str,
    host: str,
    timeout: float,
) -> Path:
    """Write a benchmark result to ``path`` and return the destination."""
    destination = Path(path)
    report = benchmark_result_to_dict(
        result, model=model, host=host, timeout=timeout
    )
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    destination.write_text(f"{content}\n", encoding="utf-8")
    return destination
