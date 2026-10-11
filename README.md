SonLithy Inference Profiler

Overview

SonLithy Inference Profiler is a Python toolkit for measuring and analyzing the inference performance of local Large Language Models (LLMs).

The project aims to help developers understand model latency, generation speed, and resource usage through reproducible benchmarks and structured reports.

Problem

Running a local LLM does not automatically provide enough information to understand its real-world performance. Developers need reliable measurements to compare configurations, identify performance differences, and make informed optimization decisions.

Target Audience

- Developers running local LLMs
- AI engineers and ML practitioners
- Open-source model enthusiasts
- Developers evaluating inference configurations

Core Features of the MVP

- Measure total request latency
- Measure time to first token when supported by the inference backend
- Measure generated tokens per second
- Run configurable benchmark iterations
- Support model warm-up before measurement
- Export benchmark results as JSON
- Handle connection failures, timeouts, and invalid configurations
- Include automated tests for core measurement logic

Initial Integration

The first integration will use the Ollama API.

The architecture should allow additional inference backends to be added later without rewriting the benchmark engine.

Design Principles

- Reproducible measurements
- Clear separation between benchmark logic and backend integration
- Minimal dependencies
- Explicit reporting of unavailable metrics
- Honest documentation of measurement limitations
- Automated testing and maintainable code

Out of Scope for the MVP

- Model training and fine-tuning
- Automatic GPU optimization
- Distributed benchmarking
- Cloud-based model hosting
- Complex dashboards and visualization interfaces

Project Status

Early development. Features are planned and are not considered implemented until verified by working code and tests.
