import math

import pytest

from sonlithy_profiler.metrics import (
    calculate_generation_throughput,
    calculate_summary_statistics,
    calculate_total_request_latency,
)


@pytest.mark.parametrize(
    ("started", "completed", "expected"),
    [
        (10.0, 12.5, 2.5),
        (2, 2, 0.0),
        (-4.0, -1.0, 3.0),
    ],
)
def test_calculate_total_request_latency(
    started: float, completed: float, expected: float
) -> None:
    assert calculate_total_request_latency(started, completed) == expected


@pytest.mark.parametrize(
    ("started", "completed"),
    [(None, 2.0), (1.0, None), (None, None)],
)
def test_latency_is_unavailable_when_timestamp_is_missing(
    started: float | None, completed: float | None
) -> None:
    assert calculate_total_request_latency(started, completed) is None


@pytest.mark.parametrize(
    ("started", "completed"),
    [
        (2.0, 1.0),
        (math.nan, 1.0),
        (1.0, math.inf),
        (-1e308, 1e308),
        ("1", 2.0),
        (True, 2.0),
    ],
)
def test_invalid_latency_inputs_raise_value_error(
    started: object, completed: object
) -> None:
    with pytest.raises(ValueError):
        calculate_total_request_latency(started, completed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("tokens", "duration", "expected"),
    [
        (20, 4.0, 5.0),
        (0, 2.0, 0.0),
        (3, 0.5, 6.0),
    ],
)
def test_calculate_generation_throughput(
    tokens: int, duration: float, expected: float
) -> None:
    assert calculate_generation_throughput(tokens, duration) == expected


@pytest.mark.parametrize(
    ("tokens", "duration"),
    [(None, 2.0), (10, None), (None, None)],
)
def test_throughput_is_unavailable_when_measurement_is_missing(
    tokens: int | None, duration: float | None
) -> None:
    assert calculate_generation_throughput(tokens, duration) is None


@pytest.mark.parametrize(
    ("tokens", "duration"),
    [
        (-1, 2.0),
        (1.5, 2.0),
        (True, 2.0),
        (1, 0),
        (1, -1.0),
        (1, math.nan),
        (1, math.inf),
        (1, "2"),
        (10**1000, 2.0),
        (10**308, 1e-308),
    ],
)
def test_invalid_throughput_inputs_raise_value_error(
    tokens: object, duration: object
) -> None:
    with pytest.raises(ValueError):
        calculate_generation_throughput(tokens, duration)  # type: ignore[arg-type]


def test_calculate_summary_statistics() -> None:
    assert calculate_summary_statistics([1, 2, 3, 4]) == {
        "mean": 2.5,
        "median": 2.5,
        "minimum": 1.0,
        "maximum": 4.0,
    }
    assert calculate_summary_statistics([1, 2, 9]) == {
        "mean": 4.0,
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 9.0,
    }


def test_summary_statistics_remain_finite_for_large_finite_values() -> None:
    assert calculate_summary_statistics([1e308, 1e308]) == {
        "mean": 1e308,
        "median": 1e308,
        "minimum": 1e308,
        "maximum": 1e308,
    }
    assert calculate_summary_statistics([1e308, 1.7e308]) == {
        "mean": 1.35e308,
        "median": 1.35e308,
        "minimum": 1e308,
        "maximum": 1.7e308,
    }


def test_summary_excludes_missing_measurements() -> None:
    assert calculate_summary_statistics([1.0, None, 3.0]) == {
        "mean": 2.0,
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
    }


@pytest.mark.parametrize("measurements", [[], [None], iter([None, None])])
def test_empty_summary_returns_unavailable_values(
    measurements: list[None] | object,
) -> None:
    assert calculate_summary_statistics(measurements) == {
        "mean": None,
        "median": None,
        "minimum": None,
        "maximum": None,
    }


@pytest.mark.parametrize(
    "measurements",
    [
        [-1.0],
        [math.nan],
        [math.inf],
        ["1"],
        [True],
    ],
)
def test_invalid_summary_measurements_raise_value_error(
    measurements: list[object],
) -> None:
    with pytest.raises(ValueError):
        calculate_summary_statistics(measurements)  # type: ignore[arg-type]
