#!/usr/bin/env python3
# File: ./tests/scitex_ml/optim/test__get_set.py

"""Tests for scitex_ml.optim._get_set module (deprecated functions)."""

import warnings

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim

from scitex_ml.optim import RANGER_AVAILABLE, get, set


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_model():
    """A simple multi-layer neural network model with learnable parameters."""
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))


@pytest.fixture
def model_list():
    """A list of independent torch models."""
    return [nn.Linear(10, 5), nn.Linear(5, 1)]


# ============================================================================
# get() returns correct optimizer class
# ============================================================================


def test_get_adam_returns_optimizer_class():
    """get('adam') returns torch.optim.Adam."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get("adam")
    # Assert
    assert result == optim.Adam


def test_get_sgd_returns_optimizer_class():
    """get('sgd') returns torch.optim.SGD."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get("sgd")
    # Assert
    assert result == optim.SGD


def test_get_rmsprop_returns_optimizer_class():
    """get('rmsprop') returns torch.optim.RMSprop."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get("rmsprop")
    # Assert
    assert result == optim.RMSprop


# ============================================================================
# get() with invalid name
# ============================================================================


def test_get_invalid_optimizer_name_raises_value_error():
    """get('invalid_optimizer') raises ValueError."""
    # Arrange
    # Act
    ctx = pytest.raises(ValueError, match="Unknown optimizer")
    # Assert
    with ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            get("invalid_optimizer")


# ============================================================================
# set() returns correct optimizer instance
# ============================================================================


def test_set_adam_returns_optimizer_instance(simple_model):
    """set(model, 'adam', lr) returns an Adam optimizer."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        optimizer = set(simple_model, "adam", 0.001)
    # Assert
    assert isinstance(optimizer, optim.Adam)


def test_set_adam_sets_correct_learning_rate(simple_model):
    """set(model, 'adam', lr) sets the learning rate in optimizer.defaults."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        optimizer = set(simple_model, "adam", 0.001)
    # Assert
    assert optimizer.defaults["lr"] == 0.001


def test_set_sgd_on_model_list_returns_optimizer(model_list):
    """set(model_list, 'sgd', lr) returns an SGD optimizer."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        optimizer = set(model_list, "sgd", 0.01)
    # Assert
    assert isinstance(optimizer, optim.SGD)


# ============================================================================
# set() with model that has no learnable parameters
# ============================================================================


def test_set_model_with_no_params_raises_value_error():
    """set() on a model with zero learnable parameters raises ValueError."""
    # Arrange
    class NoParamModel(nn.Module):
        def forward(self, x):
            return x

    model = NoParamModel()
    # Act
    ctx = pytest.raises(ValueError, match="optimizer got an empty parameter list")
    # Assert
    with ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            set(model, "adam", 0.001)


# ============================================================================
# get() deprecation warning
# ============================================================================


def test_get_adam_emits_deprecation_warning():
    """get('adam') emits a DeprecationWarning."""
    # Arrange
    # Act
    ctx = pytest.warns(DeprecationWarning)
    # Assert
    with ctx:
        get("adam")


def test_get_adam_emits_exactly_one_warning():
    """get('adam') emits exactly one warning."""
    # Arrange
    # Act
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get("adam")
    # Assert
    assert len(w) == 1


def test_get_adam_warning_message_mentions_deprecation():
    """get('adam') deprecation warning message contains 'deprecated'."""
    # Arrange
    # Act
    ctx = pytest.warns(DeprecationWarning, match="deprecated")
    # Assert
    with ctx:
        get("adam")


def test_get_adam_warning_message_mentions_get_optimizer():
    """get('adam') deprecation warning message contains 'get_optimizer'."""
    # Arrange
    # Act
    ctx = pytest.warns(DeprecationWarning, match="get_optimizer")
    # Assert
    with ctx:
        get("adam")


# ============================================================================
# set() deprecation warning
# ============================================================================


def test_set_adam_emits_deprecation_warning(simple_model):
    """set(model, 'adam', lr) emits a DeprecationWarning."""
    # Arrange
    # Act
    ctx = pytest.warns(DeprecationWarning)
    # Assert
    with ctx:
        set(simple_model, "adam", 0.001)


def test_set_adam_emits_exactly_one_warning(simple_model):
    """set(model, 'adam', lr) emits exactly one warning."""
    # Arrange
    # Act
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        set(simple_model, "adam", 0.001)
    # Assert
    assert len(w) == 1


def test_set_adam_warning_message_mentions_deprecation(simple_model):
    """set(model, 'adam', lr) deprecation warning message contains 'deprecated'."""
    # Arrange
    # Act
    ctx = pytest.warns(DeprecationWarning, match="deprecated")
    # Assert
    with ctx:
        set(simple_model, "adam", 0.001)


def test_set_adam_warning_message_mentions_set_optimizer(simple_model):
    """set(model, 'adam', lr) deprecation warning message contains 'set_optimizer'."""
    # Arrange
    # Act
    ctx = pytest.warns(DeprecationWarning, match="set_optimizer")
    # Assert
    with ctx:
        set(simple_model, "adam", 0.001)


# ============================================================================
# set() with various learning rates
# ============================================================================


@pytest.mark.parametrize("lr", [1e-4, 1e-3, 1e-2])
def test_set_adam_with_various_learning_rates(simple_model, lr):
    """set(model, 'adam', lr) sets the correct learning rate for various values."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        optimizer = set(simple_model, "adam", lr)
    # Assert
    assert optimizer.defaults["lr"] == lr


# ============================================================================
# Ranger availability
# ============================================================================


@pytest.mark.skipif(not RANGER_AVAILABLE, reason="Ranger not available")
def test_get_ranger_returns_optimizer_when_available():
    """get('ranger') returns a non-None optimizer class when Ranger is available."""
    # Arrange
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get("ranger")
    # Assert
    assert result is not None


@pytest.mark.skipif(RANGER_AVAILABLE, reason="Ranger is available")
def test_get_ranger_raises_import_error_when_unavailable():
    """get('ranger') raises ImportError when Ranger is not available."""
    # Arrange
    # Act
    ctx = pytest.raises(ImportError, match="Ranger optimizer not available")
    # Assert
    with ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            get("ranger")


# ============================================================================
# Callability
# ============================================================================


def test_get_function_is_callable():
    """get is a callable function object."""
    # Arrange
    # Act
    # Assert
    assert callable(get)


def test_set_function_is_callable():
    """set is a callable function object."""
    # Arrange
    # Act
    # Assert
    assert callable(set)


# ============================================================================
# Identity — functions importable from _get_set module
# ============================================================================


def test_get_via_package_import_is_same_object():
    """get imported directly from optim is the same object as the public API."""
    # Arrange
    from scitex_ml.optim import get as get_direct
    # Act
    # Assert
    assert get_direct is get


def test_set_via_package_import_is_same_object():
    """set imported directly from optim is the same object as the public API."""
    # Arrange
    from scitex_ml.optim import set as set_direct
    # Act
    # Assert
    assert set_direct is set
