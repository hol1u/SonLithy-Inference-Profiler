from sonlithy_profiler import __version__, benchmark, config, metrics
from sonlithy_profiler.backends import ollama
from sonlithy_profiler.reporting import json_report


def test_package_modules_are_importable() -> None:
    assert __version__ == "0.1.0"
    modules = (benchmark, config, metrics, ollama, json_report)
    assert [module.__name__ for module in modules] == [
        "sonlithy_profiler.benchmark",
        "sonlithy_profiler.config",
        "sonlithy_profiler.metrics",
        "sonlithy_profiler.backends.ollama",
        "sonlithy_profiler.reporting.json_report",
    ]
