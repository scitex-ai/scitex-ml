"""Smoke import for src/scitex_ml/classification/reporters/_mixins/_plotting.py."""

import pytest

pytest.importorskip("scitex_ml")


def test_imports():
    from scitex_ml.classification.reporters._mixins import _plotting  # noqa: F401
