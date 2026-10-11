"""Independent calculations for inference performance metrics."""

from collections.abc import Iterable
from math import isfinite
from statistics import fmean
from typing import TypeAlias

Measurement: TypeAlias = int | float | None
SummaryStatistics: TypeAlias = dict[str, float | None]


def _finite_number(value: object, name: str) -> float:
    """Return a finite numeric measurement or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def calculate_total_request_latency(
    request_started_at: Measurement,
    response_completed_at: Measurement,
) -> float | None:
    """Calculate elapsed request time in seconds from timestamp measurements.

    Return ``None`` when either timestamp is unavailable. Negative elapsed
    time or non-finite/non-numeric timestamps raise ``ValueError``.
    """
    if request_started_at is None or response_completed_at is None:
        return None

    started = _finite_number(request_started_at, "request_started_at")
    completed = _finite_number(response_completed_at, "response_completed_at")
    latency = completed - started
    if not isfinite(latency):
        raise ValueError("request latency must be finite")
    if latency < 0:
        raise ValueError("response_completed_at must not precede request_started_at")
    return latency


def calculate_generation_throughput(
    generated_tokens: int | None,
    generation_duration_seconds: Measurement,
) -> float | None:
    """Calculate generated tokens per second from backend measurements.

    A missing token count or duration yields ``None``. Token counts must be
    non-negative integers, and duration must be positive and finite; invalid
    values raise ``ValueError``. Zero generated tokens over a valid duration
    is a valid measured throughput of zero.
    """
    token_count: float | None = None
    if generated_tokens is not None:
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens < 0
        ):
            raise ValueError("generated_tokens must be a non-negative integer")
        token_count = _finite_number(generated_tokens, "generated_tokens")

    duration: float | None = None
    if generation_duration_seconds is not None:
        duration = _finite_number(
            generation_duration_seconds, "generation_duration_seconds"
        )
        if duration <= 0:
            raise ValueError("generation_duration_seconds must be greater than zero")

    if token_count is None or duration is None:
        return None
    throughput = token_count / duration
    if not isfinite(throughput):
        raise ValueError("generation throughput must be finite")
    return throughput


def calculate_summary_statistics(
    measurements: Iterable[Measurement],
) -> SummaryStatistics:
    """Calculate mean, median, minimum, and maximum of available measurements.

    ``None`` entries are excluded. Negative, non-finite, and non-numeric
    measurements raise ``ValueError``. When no measurements are available,
    each statistic is ``None`` rather than a fabricated zero.
    """
    values: list[float] = []
    for measurement in measurements:
        if measurement is None:
            continue
        value = _finite_number(measurement, "measurement")
        if value < 0:
            raise ValueError("measurements must not be negative")
        values.append(value)

    if not values:
        return {"mean": None, "median": None, "minimum": None, "maximum": None}

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        middle_value = ordered[middle]
    else:
        lower, upper = ordered[middle - 1], ordered[middle]
        middle_value = lower + (upper - lower) / 2

    scale = max(values)
    mean = 0.0 if scale == 0 else scale * fmean(value / scale for value in values)
    return {
        "mean": mean,
        "median": middle_value,
        "minimum": min(values),
        "maximum": max(values),
    }
