# SonLithy Inference Profiler

## Overview

SonLithy Inference Profiler is a Python toolkit for measuring and analyzing the
inference performance of local Large Language Models (LLMs).

The project aims to help developers understand model latency, generation speed,
and resource usage through reproducible benchmarks and structured reports.

## Project Status

Early development. Features are planned and are not considered implemented
until verified by working code and tests.

Currently implemented (v0.1.0, command line only):

- Total request latency, with mean, median, min and max
- Generated tokens per second, from Ollama's reported counts and durations
- Configurable benchmark iterations and model warm-up
- JSON report export
- Handling of connection failures, timeouts and invalid configuration
- Automated tests for the measurement logic

Not implemented yet:

- Time to first token (TTFT). The code has a place for it, but the current
  non-streaming Ollama integration cannot measure it, so it is reported as
  unavailable.
- Resource usage (CPU, RAM, GPU, VRAM) monitoring
- Backends other than Ollama
- A web UI, an API, run comparison and environment metadata

These items are planned and are not available in any release.

## Problem

Running a local LLM does not automatically provide enough information to
understand its real-world performance. Developers need reliable measurements to
compare configurations, identify performance differences, and make informed
optimization decisions.

## Target Audience

- Developers running local LLMs
- AI engineers and ML practitioners
- Open-source model enthusiasts
- Developers evaluating inference configurations

## Initial Integration

The first integration uses the Ollama API. The architecture is intended to allow
additional inference backends to be added later without rewriting the benchmark
engine.

## Design Principles

- Reproducible measurements
- Clear separation between benchmark logic and backend integration
- Minimal dependencies
- Explicit reporting of unavailable metrics
- Honest documentation of measurement limitations
- Automated testing and maintainable code

## Out of Scope for the MVP

- Model training and fine-tuning
- Automatic GPU optimization
- Distributed benchmarking
- Cloud-based model hosting
- Complex dashboards and visualization interfaces

## Usage Guide

A Python command-line toolkit for measuring local LLM inference latency and
generation throughput through Ollama.

### Requirements

- Python 3.10 or newer. CI is configured to test Python 3.10-3.13, but the
  workflow has not run yet. Python 3.14 is included as an experimental CI
  version and is not yet an official support claim.
- A running [Ollama](https://ollama.com/) server
- A model downloaded into Ollama (for example, `llama3.2`)

### Install for development

From the repository root, install the package and its test dependencies in
editable mode:

```bash
python -m pip install -e ".[dev]"
```

### Start Ollama

Start the local Ollama server in a separate terminal if it is not already
running:

```bash
ollama serve
```

Download a model before benchmarking it:

```bash
ollama pull llama3.2
```

The default server address is `http://localhost:11434`.

### Run a benchmark

```bash
sonlithy-profiler benchmark \
  --model llama3.2 \
  --prompt "Explain what an inference benchmark measures." \
  --iterations 5 \
  --warmup 1 \
  --host http://localhost:11434 \
  --timeout 60 \
  --output results.json
```

`--iterations` defaults to 1, `--warmup` to 0, `--host` to
`http://localhost:11434`, and `--timeout` to 60 seconds. `--output` is optional.
The command also supports `--help` and `--version`.

Example terminal output:

```text
Model: llama3.2
Iterations: 5 successful, 0 failed
Latency: mean 1.234 s, median 1.180 s
Throughput: 42.500 tokens/s
TTFT: N/A
JSON report: results.json
```

Latency is total request wall-clock time; throughput uses Ollama's reported
generated-token count and generation duration. A metric unavailable from the
backend is shown as `N/A` in the terminal and `null` in JSON, never as a
fabricated zero. The current non-streaming Ollama integration does not expose
the time of the first generated token, so TTFT is unavailable and is not
estimated from total request latency. Failed measured iterations and their
errors are printed and included in the JSON report; any failed measured or
warm-up iteration makes the command exit with a non-zero status.
The JSON report includes the full prompt in its configuration, so protect the
report as you would any file containing prompt data. The report records
measured iterations and warm-up failure reasons; successful warm-up measurements
are excluded from the measured statistics.

See [`examples/sample-report.json`](examples/sample-report.json) for a small,
sanitized example of the report format. It is illustrative sample data: it uses
a generic prompt and representative measurements rather than a real local run.

TTFT, CPU/GPU utilization and memory, cross-run comparisons, and automatic
environment metadata collection are not implemented in this release.
Generation settings other than the selected model and prompt cannot be
configured; reports do not collect the machine or model-runtime metadata needed
to explain hardware-specific results or warn about incomparable runs.

### Tests

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

## License

Released under the [MIT License](LICENSE).
