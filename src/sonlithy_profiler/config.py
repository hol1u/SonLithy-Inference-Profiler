"""Configuration definitions for SonLithy Inference Profiler."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by the benchmark engine and inference backends."""

    prompt: str
    iterations: int = 1
    warmup_iterations: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        if (
            isinstance(self.warmup_iterations, bool)
            or not isinstance(self.warmup_iterations, int)
            or self.warmup_iterations < 0
        ):
            raise ValueError("warmup_iterations must be a non-negative integer")
