#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/test_mcp_server.py
# ----------------------------------------
"""Tests for the scitex-ml MCP server (bare-named tools, umbrella-mounted as ml_*).

No mocks; the FastMCP instance and its tool callables run for real. One
assertion per test, AAA markers, descriptive names (TQ002/TQ003/TQ007).
"""

import asyncio

import pandas as pd
import pytest

from scitex_ml import _mcp_server


@pytest.fixture
def predictions_csv(tmp_path) -> str:
    df = pd.DataFrame({"y_true": [0, 1, 0, 1], "y_pred": [0, 1, 0, 0]})
    path = tmp_path / "preds.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_server_registers_the_five_analysis_and_skills_tools():
    # Arrange
    expected = {
        "compute_metrics",
        "generate_report",
        "reduce_dimensions",
        "skills_list",
        "skills_get",
    }
    # Act
    names = {t.name for t in asyncio.run(_mcp_server.mcp.list_tools())}
    # Assert
    assert expected == names


def test_compute_metrics_tool_reports_success_true(predictions_csv):
    # Arrange
    predictions = predictions_csv
    # Act
    out = _mcp_server.compute_metrics(predictions)
    # Assert
    assert out["success"] is True


def test_compute_metrics_tool_reports_failure_on_bad_path():
    # Arrange
    missing = "/nonexistent/preds.csv"
    # Act
    out = _mcp_server.compute_metrics(missing)
    # Assert
    assert out["success"] is False


def test_skills_list_tool_includes_installation_leaf():
    # Arrange
    expected_leaf = "01_installation"
    # Act
    out = _mcp_server.skills_list()
    # Assert
    assert expected_leaf in out["skills"]


# EOF
