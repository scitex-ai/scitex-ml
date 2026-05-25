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
_EXAMPLE_EXISTS = EXAMPLE.is_file()


@pytest.mark.skipif(not _EXAMPLE_EXISTS, reason=f"missing example: {EXAMPLE}")
def test_example_classifier_script_executes_and_exits_zero():
    # Arrange
    # Act
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True, text=True, timeout=120,
    )
    # Assert
    assert r.returncode == 0
