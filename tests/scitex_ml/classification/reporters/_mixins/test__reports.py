"""Tests for scitex_ml.classification.reporters._mixins._reports module."""

import pytest

pytest.importorskip("scitex_ml")


def test_reports_mixin_class_exists_and_has_generate_method():
    # Arrange
    from scitex_ml.classification.reporters._mixins._reports import ReportsMixin
    # Act
    has_method = hasattr(ReportsMixin, "generate_reports")
    # Assert
    assert has_method
