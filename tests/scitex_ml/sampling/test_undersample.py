"""Tests for scitex_ml.sampling module."""

import pytest

pytest.importorskip("scitex_ml")


def test_module_imports_under_expected_dotted_name():
    # Arrange / Act
    import scitex_ml.sampling.undersample
    # Assert
    assert scitex_ml.sampling.undersample is not None
