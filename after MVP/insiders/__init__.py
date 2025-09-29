"""Insiders Risk Scoring engine package.

This package provides a config-driven, modular, and testable implementation
of an insiders risk scoring engine focused on market impact from an investor's
perspective. It includes:

- Config schema and defaults
- Entity labeling and merging utilities
- Free-float computation at a fixed t0 snapshot
- Metric computations (FR, EMR, SPI)
- Event detection (penalties/reliefs)
- Normalization (percentile + robust-z with hierarchical backoff)
- Scoring and grading
- IO connectors (mock interfaces)
- FastAPI server endpoints

All network and external dependencies are mocked/skeletons to allow unit
testing without real API keys or calls.
"""

__all__ = [
    "config_schema",
    "entities",
    "freefloat",
    "metrics",
    "events",
    "normalize",
    "scorer",
]

