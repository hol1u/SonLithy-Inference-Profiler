# SonLithy Inference Profiler

A Python command-line toolkit for measuring local LLM inference latency and
generation throughput through Ollama.

## Requirements

- Python 3.10 or newer
- A running [Ollama](https://ollama.com/) server
- A model downloaded into Ollama (for example, `llama3.2`)

## Install for development

From the repository root, install the package and its test dependencies in
editable mode:

```bash
python -m pip install -e ".[dev]"
```

## Start Ollama

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

## Run a benchmark

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
sanitized example of the report format. It uses a generic prompt and
representative measurements rather than a real local run.

TTFT, CPU/GPU utilization and memory, cross-run comparisons, and automatic
environment metadata collection are not implemented in this release.
Generation settings other than the selected model and prompt cannot be
configured; reports do not collect the machine or model-runtime metadata needed
to explain hardware-specific results or warn about incomparable runs.

## Tests

```bash
python -m pytest
```
