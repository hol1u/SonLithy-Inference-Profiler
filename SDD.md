SonLithy Inference Profiler

Software Design Document (SDD)

Version: 0.1.0
Status: MVP implemented (0.1.0). This document describes the MVP; planned work beyond it is not covered here.

1. Purpose

SonLithy Inference Profiler is a Python command-line toolkit for benchmarking local Large Language Model (LLM) inference performance.

The first release focuses on reliable measurements, reproducible benchmark runs, and machine-readable JSON reports.

2. MVP Scope

The MVP must:

- Connect to a running Ollama instance.
- Send a prompt to a specified model.
- Measure total request duration.
- Measure time to first token when the backend provides sufficient timing information.
- Measure generation throughput in tokens per second.
- Support warm-up requests and multiple measured iterations.
- Aggregate measurements using mean, median, minimum, and maximum where applicable.
- Export results to JSON.
- Report errors clearly without silently producing misleading measurements.
- Include automated tests.

GPU utilization, GPU memory, CPU utilization, and system RAM monitoring are not mandatory for the first release.

3. Architecture

The project should use small, clearly separated modules.

CLI Layer

Responsibilities:

- Parse command-line arguments.
- Validate user configuration.
- Invoke the benchmark runner.
- Display concise results.
- Save reports.

Benchmark Engine

Responsibilities:

- Coordinate warm-up and measured iterations.
- Record measurement timestamps.
- Calculate latency and throughput.
- Aggregate successful measurements.
- Preserve enough configuration metadata to interpret results.

Backend Layer

Responsibilities:

- Communicate with Ollama.
- Submit inference requests.
- Parse generated text and backend-provided timing metadata.
- Normalize responses into a common internal representation.
- Report unsupported metrics explicitly.

Reporting Layer

Responsibilities:

- Serialize benchmark results to JSON.
- Include configuration, environment metadata when available, iteration results, summary statistics, and errors.
- Produce valid, readable, machine-compatible output.

4. Metric Definitions

Total Request Latency

Elapsed wall-clock time from sending a request until the complete response has been received.

Time to First Token (TTFT)

Time from request initiation until the first generated token becomes available.

TTFT must not be inferred from total request duration. If the backend integration cannot reliably expose the first-token event, report TTFT as unavailable.

Generation Throughput

Generated output tokens divided by the measured generation duration, expressed as tokens per second.

Prefer backend-reported token counts and generation duration when they are available and reliable. Document the calculation method in every report.

Do not confuse total request latency with generation duration.

5. Benchmark Procedure

1. Validate the selected model and configuration.
2. Connect to the configured Ollama endpoint.
3. Run the requested number of warm-up iterations.
4. Execute the measured iterations.
5. Record successful measurements and individual failures.
6. Calculate summary statistics from valid measurements.
7. Export a JSON report when requested.
8. Return a non-zero exit status if the benchmark cannot be completed successfully.

Warm-up measurements must not be included in the measured summary.

The report must record iteration counts and explain any excluded failures.

6. Reproducibility

Each report should include, when available:

- SonLithy version
- Backend name and endpoint
- Model identifier
- Benchmark prompt or a clearly documented prompt representation
- Generation parameters
- Warm-up count
- Requested and successful measured iteration counts
- Timestamp
- Measurement methodology
- Relevant backend and system metadata

Run comparisons and warnings about differing settings are planned but not
implemented; reports do not currently compare runs.

Results are influenced by hardware, model size, quantization, context length, generation parameters, background workloads, and backend implementation.

7. Error Handling

Handle at least:

- Backend unavailable
- Connection timeout
- Model not found
- Invalid arguments
- Invalid or incomplete backend response
- JSON report write failure
- Failed benchmark iterations

Errors must be understandable and must not be silently converted into zero-valued performance measurements.

8. Testing

Use pytest.

Tests must cover:

- Metric calculations
- Summary statistics
- Configuration validation
- Warm-up exclusion
- Backend response parsing
- Partial iteration failures
- JSON serialization
- Error handling

Unit tests must not require a running Ollama server.

Integration tests may use a real Ollama instance when explicitly configured. No such tests exist yet; the automated suite uses mocked responses only.

CI is configured to test Python 3.10-3.13, but the workflow has not run yet.

Use deterministic fixtures and mocked backend responses wherever practical.

9. Project Structure

sonlithy-inference-profiler/
├── .gitignore
├── README.md
├── CHANGELOG.md
├── LICENSE
├── SDD.md
├── pyproject.toml
├── .github/workflows/ci.yml
├── examples/
│   └── sample-report.json
├── src/
│   └── sonlithy_profiler/
│       ├── __init__.py
│       ├── py.typed
│       ├── cli.py
│       ├── config.py
│       ├── benchmark.py
│       ├── metrics.py
│       ├── backends/
│       │   ├── __init__.py
│       │   └── ollama.py
│       └── reporting/
│           ├── __init__.py
│           └── json_report.py
└── tests/
    ├── test_benchmark.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_metrics.py
    ├── test_ollama.py
    ├── test_package_imports.py
    ├── test_package_metadata.py
    └── test_reporting.py

The implementation may adjust this structure when there is a documented technical reason.

10. Code Quality

- Use Python type hints.
- Follow PEP 8 conventions.
- Keep functions focused and reasonably small.
- Avoid unnecessary abstractions and dependencies.
- Use clear names and meaningful error messages.
- Keep business logic independent of CLI presentation.
- Use a "pyproject.toml" for project metadata and development dependencies.

11. Acceptance Criteria

The MVP is complete when:

1. A user can benchmark a selected model through Ollama.
2. Warm-up and measured iterations are separated.
3. Latency and throughput are calculated using documented methods.
4. Unsupported metrics are reported honestly.
5. Results can be exported to valid JSON.
6. Errors are handled without fabricated measurements.
7. Automated tests pass without requiring a live model server.
8. Installation and basic usage are documented in README.md.