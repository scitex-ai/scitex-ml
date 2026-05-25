"""Tests for scitex_ml.classification.reporters._mixins._metrics module."""

import pytest

pytest.importorskip("scitex_ml")
numpy = pytest.importorskip("numpy")


def test_metrics_mixin_calculate_metrics_returns_dict():
    # Arrange
    from scitex_ml.classification.reporters._mixins._metrics import MetricsMixin

    class _FakeStorage:
        def save(self, data, path):
            pass

    class _Fake(MetricsMixin):
        def __init__(self):
            self.output_dir = None
            self.fold_metrics = {}
            self.all_predictions = []
            self.session_config = {}
            self.storage = _FakeStorage()

        def _round_numeric(self, value):
            return value

        def _save_fold_metrics(self, metrics, fold, labels):
            pass

        def _create_plots(self, y_true, y_pred, y_proba, labels, fold, metrics):
            pass

    fake = _Fake()
    y_true = numpy.array([0, 1, 0, 1])
    y_pred = numpy.array([0, 1, 1, 1])
    # Act
    result = fake.calculate_metrics(y_true, y_pred, verbose=False)
    # Assert
    assert isinstance(result, dict)
