"""Smoke tests for scitex_ml — verify the package and key public symbols import.

These tests are intentionally minimal. Full unit coverage of factored-out
behaviour lives in the original tests copied from scitex-python's
`tests/scitex/ai/` (or migrated incrementally as the API stabilises).
"""

from __future__ import annotations

import pytest

# Heavy optional deps — declared in `[heavy]` extra. Skip the smoke
# suite gracefully when they're not installed (e.g. `pip install -e .[dev]`
# without `[heavy]`). PS210 — see _skills/general/
# 01_ecosystem_02_dependency-and-version-pinning.md `[dev]` extras completeness.
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("optuna")
pytest.importorskip("pytorch_pretrained_vit")


def test_import_scitex_ml():
    import scitex_ml  # noqa: F401


def test_public_api_attrs():
    import scitex_ml

    expected = {
        "ClassificationReporter",
        "Classifier",
        "EarlyStopping",
        "LearningCurveLogger",
        "MultiTaskLoss",
        "get_optimizer",
        "set_optimizer",
    }
    missing = sorted(s for s in expected if not hasattr(scitex_ml, s))
    assert not missing, f"Missing public symbols on scitex_ml: {missing}"


def test_submodules_importable():
    submods = [
        "activation",
        "classification",
        "clustering",
        "feature_extraction",
        "loss",
        "metrics",
        "optim",
        "plt",
        "sampling",
        "sklearn",
        "training",
        "utils",
    ]
    import importlib

    failures = []
    for name in submods:
        try:
            importlib.import_module(f"scitex_ml.{name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "Subpackage imports failed:\n  " + "\n  ".join(failures)
