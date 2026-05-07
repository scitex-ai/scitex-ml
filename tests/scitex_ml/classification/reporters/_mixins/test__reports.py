"""Smoke import for src/scitex_ml/classification/reporters/_mixins/_reports.py."""

import pytest

pytest.importorskip("scitex_ml")


def test_imports():
    from scitex_ml.classification.reporters._mixins import _reports  # noqa: F401
