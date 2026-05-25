"""Tests for scitex_ml.classification.reporters._mixins._metrics module."""

import pytest

pytest.importorskip("scitex_ml")
numpy = pytest.importorskip("numpy")


def test_metrics_mixin_calculate_metrics_returns_dict():
    # Arrange
    from scitex_ml.classification.reporters._mixins._metrics import MetricsMixin

    class _Fake(MetricsMixin):
        def __init__(self):
            self.output_dir = None
            self.fold_metrics = {}
            self.all_predictions = []
            self.session_config = {}

        def _round_numeric(self, value):
            return value

    fake = _Fake()
    y_true = numpy.array([0, 1, 0, 1])
    y_pred = numpy.array([0, 1, 1, 1])
    # Act
    result = fake.calculate_metrics(y_true, y_pred, verbose=False)
    # Assert
    assert isinstance(result, dict)
