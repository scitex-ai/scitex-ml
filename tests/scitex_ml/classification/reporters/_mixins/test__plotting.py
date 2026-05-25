"""Tests for scitex_ml.classification.reporters._mixins._plotting module."""

import pytest

pytest.importorskip("scitex_ml")
numpy = pytest.importorskip("numpy")


def test_plotting_mixin_create_plots_writes_files_to_tmp_path(tmp_path):
    # Arrange
    from scitex_ml.classification.reporters._mixins._plotting import PlottingMixin

    class _Fake(PlottingMixin):
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.fold_metrics = {}
            self.all_predictions = []
            self.session_config = {}

        def _create_subdir_if_needed(self, name):
            from pathlib import Path
            p = Path(self.output_dir) / name
            p.mkdir(parents=True, exist_ok=True)
            return p

    fake = _Fake(tmp_path)
    y_true = numpy.array([0, 1, 0, 1])
    y_pred = numpy.array([0, 1, 1, 1])
    y_proba = numpy.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
    labels = ["A", "B"]
    # Act
    fake._create_plots(y_true, y_pred, y_proba, labels, fold=0, metrics={})
    # Assert
    saved_files = list(tmp_path.rglob("*"))
    assert len(saved_files) > 0
