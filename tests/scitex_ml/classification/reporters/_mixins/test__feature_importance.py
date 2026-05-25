"""Tests for scitex_ml.classification.reporters._mixins._feature_importance module."""

import pytest

pytest.importorskip("scitex_ml")


def test_feature_importance_mixin_class_exists_and_has_save_method():
    # Arrange
    from scitex_ml.classification.reporters._mixins._feature_importance import (
        FeatureImportanceMixin,
    )
    # Act
    has_method = hasattr(FeatureImportanceMixin, "save_feature_importance")
    # Assert
    assert has_method
