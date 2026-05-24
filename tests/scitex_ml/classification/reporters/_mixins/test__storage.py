"""Tests for scitex_ml.classification.reporters._mixins._storage module."""

import pytest

pytest.importorskip("scitex_ml")
pytest.importorskip("numpy")


def test_storage_mixin_save_scalar_metric_creates_file_in_tmp_path(tmp_path):
    # Arrange
    from scitex_ml.classification.reporters._mixins._storage import StorageMixin

    class _Fake(StorageMixin):
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.fold_metrics = {}
            self.all_predictions = []
            self.session_config = {}

    fake = _Fake(tmp_path)
    # Act
    fake._save_scalar_metric(0.85, "balanced-accuracy", 0.85, fold=0, fold_dir="fold_00")
    # Assert
    saved_files = list(tmp_path.rglob("*"))
    assert len(saved_files) > 0
