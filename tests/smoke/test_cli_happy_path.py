#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke: fast subprocess CLI happy-path (<60s, runs on every PR).

Drives the installed `scitex-ml` entry point in a child interpreter via
`python -m scitex_ml` (sys.executable, so the venv under test is used
regardless of PATH). No mocks; one assertion per test with AAA markers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_CMD = [sys.executable, "-m", "scitex_ml"]


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a subprocess with a hard timeout (never hang CI)."""
    return subprocess.run(
        [*_CMD, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def predictions_csv(tmp_path: Path) -> str:
    """Write a tiny predictions table the CLI can score."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 1, 1, 1],
            "y_proba": [0.2, 0.8, 0.6, 0.7],
        }
    )
    path = tmp_path / "preds.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_smoke_help_exits_zero(tmp_path):
    """`scitex-ml --help` exits 0 in a subprocess."""
    # Arrange
    # Act
    result = _run("--help", cwd=tmp_path)
    # Assert
    assert result.returncode == 0


def test_smoke_help_mentions_compute_metrics(tmp_path):
    """`scitex-ml --help` advertises the compute-metrics verb."""
    # Arrange
    # Act
    result = _run("--help", cwd=tmp_path)
    # Assert
    assert "compute-metrics" in result.stdout


def test_smoke_compute_metrics_exits_zero(predictions_csv, tmp_path):
    """`compute-metrics preds.csv --json` exits 0 on a valid table."""
    # Arrange
    # Act
    result = _run("compute-metrics", predictions_csv, "--json", cwd=tmp_path)
    # Assert
    assert result.returncode == 0


def test_smoke_compute_metrics_reports_balanced_accuracy(
    predictions_csv, tmp_path
):
    """`compute-metrics --json` payload carries balanced_accuracy."""
    # Arrange
    # Act
    result = _run("compute-metrics", predictions_csv, "--json", cwd=tmp_path)
    # Assert
    assert "balanced_accuracy" in json.loads(result.stdout)["metrics"]


# EOF
