"""Tests for scitex_ml.feature_extraction module."""

import pytest

pytest.importorskip("scitex_ml")


def test_module_imports_under_expected_dotted_name():
    # Arrange
    # Act
    import scitex_ml.feature_extraction
    # Assert
    assert scitex_ml.feature_extraction is not None
