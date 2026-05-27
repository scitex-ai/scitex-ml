#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/test_cli.py
# ----------------------------------------
"""Tests for the scitex-ml CLI surface (click group, in-process via CliRunner).

No mocks; commands run against real temp files. One assertion per test, AAA
markers, descriptive names (TQ002/TQ003/TQ007).
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from scitex_ml._cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def predictions_csv(tmp_path: Path) -> str:
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


def test_main_help_lists_analysis_category(runner):
    # Arrange
    args = ["--help"]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert "Analysis" in result.output


def test_version_flag_prints_package_slash_version(runner):
    # Arrange
    args = ["--version"]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert result.output.startswith("scitex-ml/") or "/" in result.output


def test_compute_metrics_json_emits_balanced_accuracy_field(runner, predictions_csv):
    # Arrange
    args = ["compute-metrics", predictions_csv, "--json"]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert "balanced_accuracy" in json.loads(result.output)["metrics"]


def test_compute_metrics_missing_file_exits_two(runner, tmp_path):
    # Arrange
    args = ["compute-metrics", str(tmp_path / "nope.csv")]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert result.exit_code == 2


def test_generate_report_dry_run_writes_nothing(runner, predictions_csv, tmp_path):
    # Arrange
    out_dir = tmp_path / "rep"
    args = ["generate-report", predictions_csv, "-o", str(out_dir), "--dry-run"]
    # Act
    runner.invoke(main, args)
    # Assert
    assert not out_dir.exists()


def test_list_python_apis_json_returns_module_scitex_ml(runner):
    # Arrange
    args = ["list-python-apis", "--json"]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert json.loads(result.output)["module"] == "scitex_ml"


def test_mcp_list_tools_json_reports_five_tools(runner):
    # Arrange
    args = ["mcp", "list-tools", "--json"]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert json.loads(result.output)["total"] == 5


def test_mcp_install_json_emits_scitex_ml_server_entry(runner):
    # Arrange
    args = ["mcp", "install", "--json"]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert "scitex-ml" in json.loads(result.output)["config"]["mcpServers"]


def test_skills_list_includes_quick_start_leaf(runner):
    # Arrange
    args = ["skills", "list"]
    # Act
    result = runner.invoke(main, args)
    # Assert
    assert "02_quick-start" in result.output


# EOF
