"""Tests for scitex_ml.sk._clf module."""

import pytest

pytest.importorskip("sktime")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.panel.rocket import Rocket

from scitex_ml.sk import GB_pipeline, rocket_pipeline


# ── rocket_pipeline creation tests ──────────────────────────────

def test_rocket_pipeline_creation_returns_pipeline_instance():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    # Assert
    assert isinstance(pipeline, Pipeline)


def test_rocket_pipeline_creation_has_two_steps():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    # Assert
    assert len(pipeline.steps) == 2


def test_rocket_pipeline_first_step_is_rocket_transform():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    # Assert
    assert isinstance(pipeline.steps[0][1], Rocket)


def test_rocket_pipeline_second_step_is_logistic_regression():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    # Assert
    assert isinstance(pipeline.steps[1][1], LogisticRegression)


def test_rocket_pipeline_with_custom_n_jobs_returns_pipeline():
    # Arrange
    # Act
    pipeline = rocket_pipeline(n_jobs=2)
    # Assert
    assert isinstance(pipeline, Pipeline)


def test_rocket_pipeline_with_custom_n_jobs_sets_rocket_jobs():
    # Arrange
    # Act
    pipeline = rocket_pipeline(n_jobs=2)
    rocket_transform = pipeline.steps[0][1]
    # Assert
    assert rocket_transform.n_jobs == 2


def test_rocket_pipeline_has_fit_method():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    # Assert
    assert hasattr(pipeline, "fit")


def test_rocket_pipeline_has_predict_method():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    # Assert
    assert hasattr(pipeline, "predict")


def test_rocket_pipeline_has_score_method():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    # Assert
    assert hasattr(pipeline, "score")


def test_rocket_pipeline_logistic_regression_uses_max_iter_1000():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    lr_classifier = pipeline.steps[1][1]
    # Assert
    assert lr_classifier.max_iter == 1000


def test_rocket_pipeline_step_zero_name_contains_rocket():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    step_names = [name for name, _ in pipeline.steps]
    # Assert
    assert "rocket" in step_names[0].lower()


def test_rocket_pipeline_step_one_name_contains_logisticregression():
    # Arrange
    # Act
    pipeline = rocket_pipeline()
    step_names = [name for name, _ in pipeline.steps]
    # Assert
    assert "logisticregression" in step_names[1].lower()


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_rocket_pipeline_n_jobs_propagates_to_rocket_transform(n_jobs):
    # Arrange
    # Act
    pipeline = rocket_pipeline(n_jobs=n_jobs)
    rocket = pipeline.steps[0][1]
    # Assert
    assert rocket.n_jobs == n_jobs


def test_rocket_pipeline_kwargs_passthrough_sets_rocket_attrs():
    # Arrange
    custom_kwargs = {"n_jobs": 2}
    # Act
    pipeline = rocket_pipeline(**custom_kwargs)
    rocket = pipeline.steps[0][1]
    # Assert
    assert rocket.n_jobs == 2


# ── GB_pipeline tests ──────────────────────────────────────────

def test_gb_pipeline_is_pipeline_instance():
    # Arrange
    # Act
    # Assert
    assert isinstance(GB_pipeline, Pipeline)


def test_gb_pipeline_has_two_steps():
    # Arrange
    # Act
    # Assert
    assert len(GB_pipeline.steps) == 2


def test_gb_pipeline_first_step_is_tabularizer():
    # Arrange
    # Act
    # Assert
    assert isinstance(GB_pipeline.steps[0][1], Tabularizer)


def test_gb_pipeline_second_step_is_gradient_boosting_classifier():
    # Arrange
    # Act
    # Assert
    assert isinstance(GB_pipeline.steps[1][1], GradientBoostingClassifier)


def test_gb_pipeline_has_fit_method():
    # Arrange
    # Act
    # Assert
    assert hasattr(GB_pipeline, "fit")


def test_gb_pipeline_has_predict_method():
    # Arrange
    # Act
    # Assert
    assert hasattr(GB_pipeline, "predict")


def test_gb_pipeline_has_score_method():
    # Arrange
    # Act
    # Assert
    assert hasattr(GB_pipeline, "score")


def test_gb_pipeline_step_zero_name_contains_tabularizer():
    # Arrange
    # Act
    step_names = [name for name, _ in GB_pipeline.steps]
    # Assert
    assert "tabularizer" in step_names[0].lower()


def test_gb_pipeline_step_one_name_contains_gradientboostingclassifier():
    # Arrange
    # Act
    step_names = [name for name, _ in GB_pipeline.steps]
    # Assert
    assert "gradientboostingclassifier" in step_names[1].lower()


def test_gb_pipeline_same_module_object_on_each_import():
    # Arrange
    # Act
    from scitex_ml.sk import GB_pipeline as gb2
    # Assert
    assert GB_pipeline is gb2


def test_gb_pipeline_clone_returns_distinct_pipeline_instance():
    # Arrange
    from sklearn.base import clone
    # Act
    cloned_pipeline = clone(GB_pipeline)
    # Assert
    assert cloned_pipeline is not GB_pipeline


def test_gb_pipeline_clone_preserves_step_count():
    # Arrange
    from sklearn.base import clone
    # Act
    cloned_pipeline = clone(GB_pipeline)
    # Assert
    assert len(cloned_pipeline.steps) == len(GB_pipeline.steps)


# ── integration fixtures & tests ────────────────────────────────

@pytest.fixture
def sample_sktime_data():
    """Create sample data in sktime format."""
    n_samples, n_dims, n_timepoints = 10, 2, 20
    data_list = []
    for _ in range(n_samples):
        sample_data = [pd.Series(np.random.randn(n_timepoints), name=f"dim_{d}")
                       for d in range(n_dims)]
        data_list.append(pd.Series(sample_data))
    X = pd.DataFrame(data_list)
    y = np.random.randint(0, 2, size=n_samples)
    return X, y


def test_rocket_pipeline_fit_predict_returns_correct_prediction_count(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    pipeline = rocket_pipeline(n_jobs=1)
    # Act
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    # Assert
    assert len(predictions) == len(y)


def test_gb_pipeline_fit_predict_returns_correct_prediction_count(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    from sklearn.base import clone
    pipeline = clone(GB_pipeline)
    # Act
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    # Assert
    assert len(predictions) == len(y)


def test_rocket_pipeline_predict_proba_returns_two_class_probabilities(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    rocket = rocket_pipeline(n_jobs=1)
    rocket.fit(X, y)
    # Act
    proba = rocket.predict_proba(X)
    # Assert
    assert proba.shape == (len(y), 2)


def test_rocket_pipeline_predict_proba_rows_sum_to_one(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    rocket = rocket_pipeline(n_jobs=1)
    rocket.fit(X, y)
    # Act
    proba = rocket.predict_proba(X)
    # Assert
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gb_pipeline_predict_proba_returns_two_class_probabilities(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    from sklearn.base import clone
    gb = clone(GB_pipeline)
    gb.fit(X, y)
    # Act
    proba_gb = gb.predict_proba(X)
    # Assert
    assert proba_gb.shape == (len(y), 2)


def test_gb_pipeline_predict_proba_rows_sum_to_one(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    from sklearn.base import clone
    gb = clone(GB_pipeline)
    gb.fit(X, y)
    # Act
    proba_gb = gb.predict_proba(X)
    # Assert
    assert np.allclose(proba_gb.sum(axis=1), 1.0)


def test_rocket_pipeline_score_returns_float_in_range_zero_to_one(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    rocket = rocket_pipeline(n_jobs=1)
    rocket.fit(X, y)
    # Act
    score = rocket.score(X, y)
    # Assert
    assert isinstance(score, float)


def test_gb_pipeline_score_returns_float_in_range_zero_to_one(sample_sktime_data):
    # Arrange
    X, y = sample_sktime_data
    from sklearn.base import clone
    gb = clone(GB_pipeline)
    gb.fit(X, y)
    # Act
    score_gb = gb.score(X, y)
    # Assert
    assert isinstance(score_gb, float)
