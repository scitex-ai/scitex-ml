#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_mcp_server.py
# ----------------------------------------
from __future__ import annotations

"""MCP server for scitex-ml — the stateless analysis surface for AI agents.

Single ``FastMCP`` instance at the canonical ``scitex_ml._mcp_server.mcp``
location (general/03_interface/03_mcp §1). Tool names are **bare**
(``compute_metrics``, ...) — the scitex umbrella mounts this server with
``namespace="ml"`` so they surface to agents as ``ml_compute_metrics`` etc.
without double-prefixing.

Every tool delegates to :mod:`scitex_ml._analysis`, so the MCP surface and the
``scitex-ml`` CLI compute identical results with identical JSON shapes
(CLI ↔ MCP parity).

Run with::

    scitex-ml mcp start
    # or
    fastmcp run scitex_ml._mcp_server:mcp
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from scitex_ml import _analysis

mcp = FastMCP(
    name="scitex-ml",
    instructions=(
        "scitex-ml exposes the stateless ML analysis surface: turn saved "
        "predictions into classification metrics and reports, and project "
        "feature matrices to 2-D. Use compute_metrics for quick numbers, "
        "generate_report to write a full report with plots, and "
        "reduce_dimensions for a PCA/UMAP figure. Training, optimizers and "
        "early-stopping are intentionally Python-only."
    ),
)

_SKILLS_PKG = "scitex-ml"


# ----------------------------------------------------------------------
# Analysis tools — each mirrors a CLI verb and an _analysis entry point
# ----------------------------------------------------------------------
@mcp.tool()
def compute_metrics(
    predictions: str, labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute classification metrics from a saved predictions table.

    Parameters
    ----------
    predictions : str
        Path to a table with ``y_true``/``y_pred`` columns (and optional
        ``y_proba`` / ``y_proba_*`` probability columns).
    labels : list of str, optional
        Class display names, in label order.

    Returns
    -------
    dict
        ``{success, predictions, metrics: {balanced_accuracy, mcc,
        confusion_matrix, labels, [roc_auc, pr_auc]}}``.
    """
    try:
        out = _analysis.run_metrics(predictions, labels=labels)
        return {"success": True, **out}
    except _analysis.AnalysisError as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def generate_report(
    predictions: str, output_dir: str, labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate a full ClassificationReporter report from saved predictions.

    Writes a metrics summary and plots under ``output_dir``.

    Parameters
    ----------
    predictions : str
        Path to a table with ``y_true``/``y_pred`` (+ optional probabilities).
    output_dir : str
        Directory for the report artifacts (created if absent).
    labels : list of str, optional
        Class display names, in label order.

    Returns
    -------
    dict
        ``{success, predictions, output_dir, summary_path, metrics}``.
    """
    try:
        out = _analysis.run_report(predictions, output_dir, labels=labels)
        return {"success": True, **out}
    except _analysis.AnalysisError as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def reduce_dimensions(
    data: str,
    output: str,
    method: str = "pca",
    label_col: Optional[str] = None,
    supervised: bool = False,
) -> Dict[str, Any]:
    """Project a feature matrix to 2-D (PCA or UMAP) and save a figure.

    Parameters
    ----------
    data : str
        Path to a table of features; one row per sample.
    output : str
        Path for the saved figure (extension picks the format).
    method : {"pca", "umap"}
        Reduction method. UMAP needs the optional ``umap-learn`` dependency.
    label_col : str, optional
        Column to colour points by (split out of the feature matrix).
    supervised : bool
        UMAP only — fit using the labels.

    Returns
    -------
    dict
        ``{success, method, figure_path, n_samples, n_features}``.
    """
    try:
        out = _analysis.run_reduce(
            data,
            output,
            method=method,
            label_col=label_col,
            supervised=supervised,
        )
        return {"success": True, **out}
    except _analysis.AnalysisError as e:
        return {"success": False, "error": str(e)}


# ----------------------------------------------------------------------
# Skills introspection tools (mandatory per general/03_interface/03_mcp §5)
# Bare names → `ml_skills_list` / `ml_skills_get` once mounted by the umbrella.
# ----------------------------------------------------------------------
def _skills_root() -> Path:
    import scitex_ml

    return Path(scitex_ml.__file__).parent / "_skills" / _SKILLS_PKG


def _skill_files() -> List[Path]:
    root = _skills_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file() and p.name != "SKILL.md")


@mcp.tool()
def skills_list() -> Dict[str, Any]:
    """List the agent-facing skill pages bundled with scitex-ml."""
    return {
        "success": True,
        "package": _SKILLS_PKG,
        "skills": [p.stem for p in _skill_files()],
    }


@mcp.tool()
def skills_get(name: str) -> Dict[str, Any]:
    """Fetch the full Markdown content of one scitex-ml skill page by name."""
    stem = name[:-3] if name.endswith(".md") else name
    match = next((p for p in _skill_files() if p.stem == stem), None)
    if match is None:
        return {
            "success": False,
            "error": f"unknown skill {name!r}",
            "available": [p.stem for p in _skill_files()],
        }
    return {
        "success": True,
        "package": _SKILLS_PKG,
        "name": match.stem,
        "content": match.read_text(encoding="utf-8"),
    }


if __name__ == "__main__":  # pragma: no cover
    mcp.run()


# EOF
