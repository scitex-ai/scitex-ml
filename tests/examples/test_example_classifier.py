"""Smoke test for examples/example_classifier.py — runs the script and checks exit 0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Heavy optional deps pulled in by `scitex_ml.classification`.
pytest.importorskip("torch")
pytest.importorskip("sklearn")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "example_classifier.py"


def test_example_classifier_runs():
    assert EXAMPLE.is_file(), f"missing example: {EXAMPLE}"
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
