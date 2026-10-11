# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The baseline is the MVP at commit `15e84fc` (version 0.1.0, not yet released).

## [Unreleased]

### Added

- MIT `LICENSE`.
- GitHub Actions CI: tests on Python 3.10-3.13, an experimental Python 3.14
  job, Ruff, mypy, and a package build check.
- Package metadata (license, author, URLs, classifiers) and the `py.typed`
  marker.
- Ruff and mypy as development dependencies.
- A test that keeps package metadata consistent.

### Changed

- README documents the Python support policy and the license.
- SDD corrected to describe the implemented MVP.
