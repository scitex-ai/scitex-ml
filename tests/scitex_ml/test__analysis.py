#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/test_analysis_surface.py
# ----------------------------------------
"""Tests for the stateless analysis core shared by the CLI and MCP surfaces.

Real data, no mocks (general/02_package §12). One assertion per test, AAA
markers, descriptive names (TQ002/TQ003/TQ007).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scitex_ml import _analysis


@pytest.fixture
def predictions_csv(tmp_path: Path) -> str:
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 0, 1],
            "y_pred": [0, 1, 1, 1, 0, 0],
            "y_proba": [0.1, 0.9, 0.6, 0.8, 0.2, 0.45],
        }
    )
    path = tmp_path / "preds.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def features_csv(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (15, 4)), rng.normal(5, 1, (15, 4))])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["label"] = [0] * 15 + [1] * 15
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_run_metrics_returns_balanced_accuracy_between_zero_and_one(
    predictions_csv,
):
    # Arrange
    predictions = predictions_csv
    # Act
    out = _analysis.run_metrics(predictions)
    # Assert
    assert 0.0 <= out["metrics"]["balanced_accuracy"] <= 1.0


def test_run_metrics_includes_roc_auc_when_probabilities_present(
    predictions_csv,
):
    # Arrange
    predictions = predictions_csv
    # Act
    out = _analysis.run_metrics(predictions)
    # Assert
    assert "roc_auc" in out["metrics"]


def test_run_metrics_confusion_matrix_is_nested_list_of_lists(
    predictions_csv,
):
    # Arrange
    predictions = predictions_csv
    # Act
    cm = _analysis.run_metrics(predictions)["metrics"]["confusion_matrix"]
    # Assert
    assert isinstance(cm, list) and isinstance(cm[0], list)


def test_run_metrics_raises_analysis_error_on_missing_columns(tmp_path):
    # Arrange
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(bad, index=False)
    # Act
    call = lambda: _analysis.run_metrics(str(bad))
    # Assert
    with pytest.raises(_analysis.AnalysisError):
        call()


def test_run_report_writes_summary_file_to_output_dir(predictions_csv, tmp_path):
    # Arrange
    out_dir = tmp_path / "report"
    # Act
    out = _analysis.run_report(predictions_csv, str(out_dir))
    # Assert
    assert Path(out["summary_path"]).is_file()


def test_run_reduce_pca_writes_figure_file(features_csv, tmp_path):
    # Arrange
    fig_path = tmp_path / "pca.png"
    # Act
    out = _analysis.run_reduce(
        features_csv, str(fig_path), method="pca", label_col="label"
    )
    # Assert
    assert Path(out["figure_path"]).is_file()


def test_run_reduce_rejects_unknown_method_with_analysis_error(features_csv, tmp_path):
    # Arrange
    fig_path = tmp_path / "x.png"
    # Act
    call = lambda: _analysis.run_reduce(features_csv, str(fig_path), method="tsne")
    # Assert
    with pytest.raises(_analysis.AnalysisError):
        call()


# EOF
