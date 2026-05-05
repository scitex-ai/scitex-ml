"""Smoke tests for scitex_ai — verify the package and key public symbols import.

These tests are intentionally minimal. Full unit coverage of factored-out
behaviour lives in the original tests copied from scitex-python's
`tests/scitex/ai/` (or migrated incrementally as the API stabilises).
"""

from __future__ import annotations


def test_import_scitex_ai():
    import scitex_ai  # noqa: F401


def test_public_api_attrs():
    import scitex_ai

    expected = {
        "ClassificationReporter",
        "Classifier",
        "EarlyStopping",
        "LearningCurveLogger",
        "MultiTaskLoss",
        "get_optimizer",
        "set_optimizer",
        # GenAI is lazy — accessing it should not raise even if anthropic et al.
        # are not configured. Just verify the attribute lookup hits __getattr__.
        "GenAI",
    }
    missing = sorted(s for s in expected if not hasattr(scitex_ai, s))
    assert not missing, f"Missing public symbols on scitex_ai: {missing}"


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
            importlib.import_module(f"scitex_ai.{name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "Subpackage imports failed:\n  " + "\n  ".join(failures)
