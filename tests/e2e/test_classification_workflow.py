#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2E: Iris train -> report workflow against real subsystems.

Slow (model fitting + figure export), loopback-only (no network), gated
by `RUN_E2E=1` and skipped by default. One assertion per test with AAA
markers; the shared flow lives in a helper, not a fixture, so each test
reads as Arrange/Act/Assert.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("RUN_E2E") != "1",
        reason="slow e2e: set RUN_E2E=1 to run",
    ),
]


def _run_iris_workflow(output_dir: Path):
    """Fit LogisticRegression on Iris and persist a reporter summary."""
    import scitex_ml
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, stratify=y)
    model = scitex_ml.Classifier()("LogisticRegression")
    model.fit(X_tr, y_tr)
    reporter = scitex_ml.ClassificationReporter(output_dir=str(output_dir))
    metrics = reporter.calculate_metrics(
        y_te,
        model.predict(X_te),
        model.predict_proba(X_te),
        verbose=False,
    )
    summary_path = reporter.save_summary(verbose=False)
    return metrics, summary_path


def test_e2e_iris_summary_file_written(tmp_path):
    """Iris workflow persists a summary.json artefact on disk."""
    # Arrange
    # Act
    _, summary_path = _run_iris_workflow(tmp_path / "report")
    # Assert
    assert Path(summary_path).is_file()


def test_e2e_iris_balanced_accuracy_above_ninety_percent(tmp_path):
    """Iris workflow reaches >0.9 balanced-accuracy (sanity, not tuning)."""
    # Arrange
    # Act
    metrics, _ = _run_iris_workflow(tmp_path / "report")
    # Assert
    assert metrics["balanced-accuracy"]["value"] > 0.9


# EOF
