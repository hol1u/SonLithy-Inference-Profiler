import pytest

from sonlithy_profiler.config import BenchmarkConfig


def test_benchmark_config_defaults_and_accepts_valid_values() -> None:
    assert BenchmarkConfig("prompt") == BenchmarkConfig(
        "prompt", iterations=1, warmup_iterations=0
    )
    assert BenchmarkConfig("prompt", iterations=3, warmup_iterations=2)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"prompt": ""}, "prompt"),
        ({"prompt": "  "}, "prompt"),
        ({"prompt": 1}, "prompt"),
        ({"prompt": "prompt", "iterations": 0}, "iterations"),
        ({"prompt": "prompt", "iterations": -1}, "iterations"),
        ({"prompt": "prompt", "iterations": True}, "iterations"),
        ({"prompt": "prompt", "iterations": 1.5}, "iterations"),
        ({"prompt": "prompt", "warmup_iterations": -1}, "warmup_iterations"),
        ({"prompt": "prompt", "warmup_iterations": True}, "warmup_iterations"),
        ({"prompt": "prompt", "warmup_iterations": 1.5}, "warmup_iterations"),
    ],
)
def test_invalid_benchmark_configuration_raises_value_error(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BenchmarkConfig(**arguments)  # type: ignore[arg-type]
