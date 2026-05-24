"""Tests for scitex_ml.classification.reporters._mixins._cv_summary module."""

import pytest

pytest.importorskip("scitex_ml")


def test_cv_summary_mixin_class_exists_and_has_curve_method():
    # Arrange
    from scitex_ml.classification.reporters._mixins._cv_summary import CVSummaryMixin
    # Act
    has_method = hasattr(CVSummaryMixin, "create_cv_summary_curves")
    # Assert
    assert has_method
