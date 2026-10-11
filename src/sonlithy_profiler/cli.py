"""Command-line entry point for SonLithy Inference Profiler."""

import argparse
import math
import sys
from collections.abc import Sequence

from sonlithy_profiler import __version__
from sonlithy_profiler.backends.ollama import OllamaBackend
from sonlithy_profiler.benchmark import BenchmarkResult, run_benchmark
from sonlithy_profiler.config import BenchmarkConfig
from sonlithy_profiler.reporting.json_report import write_json_report

PROJECT_NAME = "SonLithy Inference Profiler"


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _non_negative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer"
        ) from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def _non_empty_path(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="sonlithy-profiler",
        description=f"{PROJECT_NAME} {__version__}",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROJECT_NAME} {__version__}",
    )
    commands = parser.add_subparsers(dest="command")
    benchmark_parser = commands.add_parser(
        "benchmark", help="Run an inference benchmark against Ollama"
    )
    benchmark_parser.add_argument("--model", required=True, help="Ollama model name")
    benchmark_parser.add_argument("--prompt", required=True, help="Prompt text")
    benchmark_parser.add_argument(
        "--iterations",
        type=_positive_integer,
        default=1,
        help="Number of measured iterations (default: 1)",
    )
    benchmark_parser.add_argument(
        "--warmup",
        type=_non_negative_integer,
        default=0,
        help="Number of warm-up iterations (default: 0)",
    )
    benchmark_parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    benchmark_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=60.0,
        help="HTTP request timeout in seconds (default: 60)",
    )
    benchmark_parser.add_argument(
        "--output",
        type=_non_empty_path,
        help="Optional path to write the JSON report",
    )
    return parser


def _format_measurement(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}{unit}"


def _print_results(result: BenchmarkResult, model: str) -> None:
    latency = result.summary["total_request_latency_seconds"]
    throughput = result.summary["generation_throughput_tokens_per_second"]
    ttft = result.summary["time_to_first_token_seconds"]
    print(f"Model: {model}")
    print(
        "Iterations: "
        f"{result.successful_iterations} successful, "
        f"{result.failed_iterations} failed"
    )
    print(
        "Latency: "
        f"mean {_format_measurement(latency['mean'], ' s')}, "
        f"median {_format_measurement(latency['median'], ' s')}"
    )
    print(
        "Throughput: "
        f"{_format_measurement(throughput['mean'], ' tokens/s')}"
    )
    print(f"TTFT: {_format_measurement(ttft['mean'], ' s')}")

    for iteration in result.iterations:
        if iteration.error is not None:
            print(
                f"Iteration {iteration.iteration} failed: {iteration.error}",
                file=sys.stderr,
            )
    for index, error in enumerate(result.warmup_failures, start=1):
        print(f"Warm-up iteration {index} failed: {error}", file=sys.stderr)


def _run_benchmark(arguments: argparse.Namespace) -> int:
    try:
        config = BenchmarkConfig(
            prompt=arguments.prompt,
            iterations=arguments.iterations,
            warmup_iterations=arguments.warmup,
        )
        backend = OllamaBackend(arguments.host, arguments.model, arguments.timeout)
        result = run_benchmark(config, backend)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    _print_results(result, arguments.model)
    if arguments.output:
        try:
            write_json_report(
                result,
                arguments.output,
                model=arguments.model,
                host=arguments.host,
                timeout=arguments.timeout,
            )
        except (OSError, TypeError, ValueError) as error:
            print(f"Error writing JSON report: {error}", file=sys.stderr)
            return 1
        print(f"JSON report: {arguments.output}")
    return 1 if result.failed_iterations or result.warmup_failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    if not arguments:
        print(f"{PROJECT_NAME} {__version__}")
        return 0
    parsed = parser.parse_args(arguments)
    if parsed.command == "benchmark":
        return _run_benchmark(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
