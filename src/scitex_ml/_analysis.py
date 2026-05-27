#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_analysis.py
# ----------------------------------------
from __future__ import annotations

"""Shared, stateless analysis core behind the CLI and MCP surfaces.

Both ``scitex-ml <verb>`` (CLI) and the MCP tools call these functions, so the
two interfaces stay in lock-step — same logic, same JSON shape (CLI ↔ MCP
parity, see ``general/03_interface`` §7). Everything here is file-in →
JSON/artifact-out with no in-process model state, which is exactly the slice
of scitex-ml that makes sense to expose to a terminal or an agent. Stateful,
code-driven pieces (training loops, EarlyStopping, optimizers) stay
Python-only on purpose.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "AnalysisError",
    "run_metrics",
    "run_report",
    "run_reduce",
]


class AnalysisError(ValueError):
    """Raised on bad input (missing columns, unknown method, ...).

    The CLI maps this to exit code 2; the MCP tools catch it and return
    ``{"success": False, "error": ...}``.
    """


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def _load_table(path: str):
    """Load a tabular file (CSV/TSV/parquet/...) into a DataFrame via scitex-io."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise AnalysisError(f"file not found: {path}")
    from scitex_io import load

    obj = load(str(p))
    if not isinstance(obj, pd.DataFrame):
        raise AnalysisError(
            f"{path}: expected a table (DataFrame), got {type(obj).__name__}"
        )
    return obj


def _load_predictions(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load ``y_true``/``y_pred`` (+ optional probabilities) from a table file.

    The file must have ``y_true`` and ``y_pred`` columns. Probabilities are
    optional and read from either a single ``y_proba`` column (binary
    positive-class probability) or one column per class named
    ``y_proba_0``, ``y_proba_1``, ... (multiclass).
    """
    df = _load_table(path)
    missing = [c for c in ("y_true", "y_pred") if c not in df.columns]
    if missing:
        raise AnalysisError(
            f"{path}: missing required column(s) {missing}; found {list(df.columns)}"
        )
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    proba_cols = sorted(c for c in df.columns if str(c).startswith("y_proba_"))
    if proba_cols:
        y_proba: Optional[np.ndarray] = df[proba_cols].to_numpy()
    elif "y_proba" in df.columns:
        y_proba = df["y_proba"].to_numpy()
    else:
        y_proba = None
    return y_true, y_pred, y_proba


def _load_features(
    path: str, label_col: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a feature matrix ``X`` (+ optional labels ``y``) from a file.

    ``label_col`` names a column to split out as the label vector; the rest
    become ``X``. With no ``label_col`` the whole table is ``X``.
    """
    df = _load_table(path)
    if label_col is not None:
        if label_col not in df.columns:
            raise AnalysisError(
                f"{path}: label column {label_col!r} not found; "
                f"columns are {list(df.columns)}"
            )
        y = df[label_col].to_numpy()
        X = df.drop(columns=[label_col]).to_numpy()
    else:
        y = None
        X = df.to_numpy()
    return X, y


# ----------------------------------------------------------------------
# Pure computation
# ----------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics — the JSON-safe core shared by all surfaces.

    Returns a dict with ``balanced_accuracy``, ``mcc``, ``confusion_matrix``
    (nested lists) and its ``labels``. When ``y_proba`` is given, ``roc_auc``
    and ``pr_auc`` are added too.
    """
    from scitex_ml import metrics as _m

    out: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "balanced_accuracy": float(_m.calc_bacc(y_true, y_pred)["value"]),
        "mcc": float(_m.calc_mcc(y_true, y_pred)["value"]),
    }

    cm = _m.calc_conf_mat(y_true, y_pred, labels=labels)["value"]
    # calc_conf_mat returns a labelled DataFrame; serialise it JSON-safe.
    out["confusion_matrix"] = cm.to_numpy().astype(int).tolist()
    out["labels"] = [str(c) for c in cm.columns.tolist()]

    if y_proba is not None:
        out["roc_auc"] = float(_m.calc_roc_auc(y_true, y_proba)["value"])
        out["pr_auc"] = float(_m.calc_pre_rec_auc(y_true, y_proba)["value"])
    return out


# ----------------------------------------------------------------------
# Verb entry points (one per CLI subcommand / MCP tool)
# ----------------------------------------------------------------------
def run_metrics(predictions: str, labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute metrics from a saved predictions table — no files written.

    Parameters
    ----------
    predictions : str
        Path to a table with ``y_true``/``y_pred`` (+ optional probabilities).
    labels : list of str, optional
        Display names for the classes, in label order.

    Returns
    -------
    dict
        ``{"predictions", "metrics": {...}}`` — see :func:`compute_metrics`.
    """
    y_true, y_pred, y_proba = _load_predictions(predictions)
    return {
        "predictions": str(Path(predictions).resolve()),
        "metrics": compute_metrics(y_true, y_pred, y_proba, labels=labels),
    }


def run_report(
    predictions: str,
    output_dir: str,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run :class:`~scitex_ml.ClassificationReporter` on saved predictions.

    Writes the full metrics summary and plots under ``output_dir`` and returns
    the computed metrics plus the artifact paths.

    Parameters
    ----------
    predictions : str
        Path to a table with ``y_true``/``y_pred`` (+ optional probabilities).
    output_dir : str
        Directory for the report artifacts (created if absent).
    labels : list of str, optional
        Display names for the classes, in label order.

    Returns
    -------
    dict
        ``{"predictions", "output_dir", "summary_path", "metrics": {...}}``.
    """
    from scitex_ml import ClassificationReporter

    y_true, y_pred, y_proba = _load_predictions(predictions)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reporter = ClassificationReporter(str(out_dir), verbose=False)
    reporter.calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
        verbose=False,
    )
    summary_path = reporter.save_summary(verbose=False)

    return {
        "predictions": str(Path(predictions).resolve()),
        "output_dir": str(out_dir.resolve()),
        "summary_path": str(Path(summary_path).resolve()),
        "metrics": compute_metrics(y_true, y_pred, y_proba, labels=labels),
    }


def run_reduce(
    data: str,
    output: str,
    method: str = "pca",
    label_col: Optional[str] = None,
    supervised: bool = False,
) -> Dict[str, Any]:
    """Project a feature matrix to 2-D with PCA or UMAP and save a figure.

    Parameters
    ----------
    data : str
        Path to a table of features; one row per sample.
    output : str
        Path for the saved figure (extension picks the format, e.g. ``.png``).
    method : {"pca", "umap"}
        Reduction method. UMAP needs the optional ``umap-learn`` dependency.
    label_col : str, optional
        Column to colour points by (split out of the feature matrix).
    supervised : bool
        UMAP only — fit using the labels (ignored for PCA).

    Returns
    -------
    dict
        ``{"method", "figure_path", "n_samples", "n_features"}``.
    """
    method = method.lower()
    if method not in ("pca", "umap"):
        raise AnalysisError(f"unknown method {method!r}; choose 'pca' or 'umap'")

    X, y = _load_features(data, label_col=label_col)
    if y is None:
        y = np.zeros(len(X), dtype=int)

    from scitex_ml import clustering as _c

    if method == "pca":
        fig, _legends, _model = _c.pca(data_all=[X], labels_all=[y])
    else:
        fig, _legends, _model = _c.umap(data=[X], labels=[y], supervised=supervised)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from scitex_io import save

    save(fig, str(out_path))

    return {
        "method": method,
        "figure_path": str(out_path.resolve()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


# EOF
