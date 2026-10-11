import re
from pathlib import Path

from sonlithy_profiler import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_version_matches_package_version() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)

    assert match is not None
    assert match.group(1) == __version__


def test_license_and_typing_marker_are_present() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License")
    assert "hol1u" in license_text
    assert (ROOT / "src" / "sonlithy_profiler" / "py.typed").is_file()
