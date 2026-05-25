#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_ml.utils.check_params.

Each test follows TQ rules: exactly one assert, standalone functions, and
# Arrange / # Act / # Assert markers on separate lines.
"""

import time

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from scitex_ml.utils import check_params


# ---------------------------------------------------------------------------
# Module-level model classes
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    """Simple test model with conv, batchnorm, linear, and dropout layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return x


class EmptyModel(nn.Module):
    """Model that has no learnable parameters."""

    def forward(self, x):
        return x


class NestedModel(nn.Module):
    """Model with nested Sequential encoder and decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 30)
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 10)
        )


class LargeModel(nn.Module):
    """Model with many linear layers for performance testing."""

    def __init__(self):
        super().__init__()
        for i in range(100):
            setattr(self, f"layer_{i}", nn.Linear(10, 10))


class SharedParamModel(nn.Module):
    """Model where multiple attributes share the same layer reference."""

    def __init__(self):
        super().__init__()
        self.shared_layer = nn.Linear(10, 10)
        self.layer1 = self.shared_layer
        self.layer2 = self.shared_layer


class CustomParamModel(nn.Module):
    """Model with directly-assigned and named registered parameters."""

    def __init__(self):
        super().__init__()
        self.custom_param = nn.Parameter(torch.randn(5, 5))
        self.register_parameter("named_param", nn.Parameter(torch.randn(3, 3)))


class BufferModel(nn.Module):
    """Model with a registered buffer alongside normal parameters."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.register_buffer("buffer", torch.randn(5, 5))


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    """Return a fresh SimpleModel instance."""
    return SimpleModel()


@pytest.fixture
def complex_model():
    """Return a SimpleModel with conv1 weights frozen."""
    model = SimpleModel()
    for param in model.conv1.parameters():
        param.requires_grad = False
    return model


@pytest.fixture
def empty_model():
    """Return a fresh EmptyModel instance."""
    return EmptyModel()


@pytest.fixture
def nested_model():
    """Return a fresh NestedModel instance."""
    return NestedModel()


@pytest.fixture
def large_model():
    """Return a fresh LargeModel instance."""
    return LargeModel()


@pytest.fixture
def shared_param_model():
    """Return a fresh SharedParamModel instance."""
    return SharedParamModel()


@pytest.fixture
def custom_param_model():
    """Return a fresh CustomParamModel instance."""
    return CustomParamModel()


@pytest.fixture
def buffer_model():
    """Return a fresh BufferModel instance."""
    return BufferModel()


# ---------------------------------------------------------------------------
# Tests: basic functionality
# ---------------------------------------------------------------------------

def test_check_params_returns_dict_for_simple_model(simple_model):
    """check_params returns a dict when given a model with parameters."""
    # Arrange
    # Act
    result = check_params(simple_model)
    # Assert
    assert isinstance(result, dict)


def test_check_params_returns_nonempty_dict_for_model_with_params(simple_model):
    """check_params returns a non-empty dict when the model has parameters."""
    # Arrange
    # Act
    result = check_params(simple_model)
    # Assert
    assert len(result) > 0


def test_check_params_entry_count_matches_named_parameter_count(simple_model):
    """check_params result has as many entries as model.named_parameters()."""
    # Arrange
    expected_params = dict(simple_model.named_parameters())
    # Act
    result = check_params(simple_model)
    # Assert
    assert len(result) == len(expected_params)


# ---------------------------------------------------------------------------
# Tests: parameter shapes
# ---------------------------------------------------------------------------

def test_check_params_result_includes_conv1_weight_key(simple_model):
    """check_params result dict includes the key 'conv1.weight'."""
    # Arrange
    # Act
    result = check_params(simple_model)
    # Assert
    assert "conv1.weight" in result


def test_check_params_conv1_weight_shape_is_16_3_3_3(simple_model):
    """check_params reports conv1.weight shape as torch.Size([16, 3, 3, 3])."""
    # Arrange
    # Act
    result = check_params(simple_model)
    shape, _ = result["conv1.weight"]
    # Assert
    assert shape == torch.Size([16, 3, 3, 3])


def test_check_params_conv1_weight_status_is_learnable(simple_model):
    """check_params reports conv1.weight status as 'Learnable'."""
    # Arrange
    # Act
    result = check_params(simple_model)
    _, status = result["conv1.weight"]
    # Assert
    assert status == "Learnable"


def test_check_params_result_includes_fc1_weight_key(simple_model):
    """check_params result dict includes the key 'fc1.weight'."""
    # Arrange
    # Act
    result = check_params(simple_model)
    # Assert
    assert "fc1.weight" in result


def test_check_params_fc1_weight_shape_matches_128_by_12544(simple_model):
    """check_params reports fc1.weight shape as torch.Size([128, 12544])."""
    # Arrange
    # Act
    result = check_params(simple_model)
    shape, _ = result["fc1.weight"]
    # Assert
    assert shape == torch.Size([128, 16 * 28 * 28])


# ---------------------------------------------------------------------------
# Tests: frozen vs learnable status (complex_model)
# ---------------------------------------------------------------------------

def test_check_params_frozen_conv1_weight_status_is_freezed(complex_model):
    """check_params reports a frozen parameter's status as 'Freezed'."""
    # Arrange
    # Act
    result = check_params(complex_model)
    _, status = result["conv1.weight"]
    # Assert
    assert status == "Freezed"


def test_check_params_learnable_fc1_weight_status_is_learnable(complex_model):
    """check_params reports a learnable parameter's status as 'Learnable'."""
    # Arrange
    # Act
    result = check_params(complex_model)
    _, status = result["fc1.weight"]
    # Assert
    assert status == "Learnable"


# ---------------------------------------------------------------------------
# Tests: target-name filtering
# ---------------------------------------------------------------------------

def test_check_params_with_target_name_returns_single_entry(simple_model):
    """check_params(simple_model, tgt_name='fc1.weight') has length 1."""
    # Arrange
    # Act
    result = check_params(simple_model, tgt_name="fc1.weight")
    # Assert
    assert len(result) == 1


def test_check_params_with_nonexistent_target_name_returns_empty_dict(
    simple_model,
):
    """check_params with a nonexistent tgt_name returns an empty dict."""
    # Arrange
    # Act
    result = check_params(simple_model, tgt_name="nonexistent.weight")
    # Assert
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: show=True prints to stdout
# ---------------------------------------------------------------------------

def test_check_params_with_show_true_produces_output(simple_model, capsys):
    """check_params(simple_model, show=True) prints something to stdout."""
    # Arrange
    # Act
    _ = check_params(simple_model, show=True)
    captured = capsys.readouterr()
    # Assert
    assert len(captured.out) > 0


def test_check_params_with_show_true_includes_conv1_weight_in_output(
    simple_model, capsys
):
    """check_params(simple_model, show=True) output mentions conv1.weight."""
    # Arrange
    # Act
    _ = check_params(simple_model, show=True)
    captured = capsys.readouterr()
    # Assert
    assert "conv1.weight" in captured.out


def test_check_params_with_show_true_includes_fc1_weight_in_output(
    simple_model, capsys
):
    """check_params(simple_model, show=True) output mentions fc1.weight."""
    # Arrange
    # Act
    _ = check_params(simple_model, show=True)
    captured = capsys.readouterr()
    # Assert
    assert "fc1.weight" in captured.out


# ---------------------------------------------------------------------------
# Tests: show with target name
# ---------------------------------------------------------------------------

def test_check_params_with_show_and_target_prints_correct_param(
    simple_model, capsys
):
    """check_params(simple_model, tgt_name='fc2.bias', show=True) prints fc2.bias."""
    # Arrange
    # Act
    _ = check_params(simple_model, tgt_name="fc2.bias", show=True)
    captured = capsys.readouterr()
    # Assert
    assert "fc2.bias" in captured.out


def test_check_params_with_show_and_target_excludes_other_params(
    simple_model, capsys
):
    """check_params with tgt_name='fc2.bias' and show=True excludes fc1.weight."""
    # Arrange
    # Act
    _ = check_params(simple_model, tgt_name="fc2.bias", show=True)
    captured = capsys.readouterr()
    # Assert
    assert "fc1.weight" not in captured.out


# ---------------------------------------------------------------------------
# Tests: empty model
# ---------------------------------------------------------------------------

def test_check_params_on_empty_model_returns_dict(empty_model):
    """check_params on an empty model returns a dict."""
    # Arrange
    # Act
    result = check_params(empty_model)
    # Assert
    assert isinstance(result, dict)


def test_check_params_on_empty_model_returns_zero_length(empty_model):
    """check_params on an empty model returns a dict with length 0."""
    # Arrange
    # Act
    result = check_params(empty_model)
    # Assert
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: nested modules
# ---------------------------------------------------------------------------

def test_check_params_includes_nested_encoder_first_weight(nested_model):
    """check_params includes key 'encoder.0.weight' for nested Sequential."""
    # Arrange
    # Act
    result = check_params(nested_model)
    # Assert
    assert "encoder.0.weight" in result


def test_check_params_includes_nested_encoder_third_weight(nested_model):
    """check_params includes key 'encoder.2.weight' for nested Sequential."""
    # Arrange
    # Act
    result = check_params(nested_model)
    # Assert
    assert "encoder.2.weight" in result


def test_check_params_includes_nested_decoder_first_weight(nested_model):
    """check_params includes key 'decoder.0.weight' for nested Sequential."""
    # Arrange
    # Act
    result = check_params(nested_model)
    # Assert
    assert "decoder.0.weight" in result


def test_check_params_includes_nested_decoder_third_weight(nested_model):
    """check_params includes key 'decoder.2.weight' for nested Sequential."""
    # Arrange
    # Act
    result = check_params(nested_model)
    # Assert
    assert "decoder.2.weight" in result


# ---------------------------------------------------------------------------
# Tests: BatchNorm parameters
# ---------------------------------------------------------------------------

def test_check_params_includes_batchnorm_weight_key(simple_model):
    """check_params result includes the BatchNorm weight key 'bn1.weight'."""
    # Arrange
    # Act
    result = check_params(simple_model)
    # Assert
    assert "bn1.weight" in result


def test_check_params_includes_batchnorm_bias_key(simple_model):
    """check_params result includes the BatchNorm bias key 'bn1.bias'."""
    # Arrange
    # Act
    result = check_params(simple_model)
    # Assert
    assert "bn1.bias" in result


def test_check_params_batchnorm_weight_shape_is_16(simple_model):
    """check_params reports bn1.weight shape as torch.Size([16])."""
    # Arrange
    # Act
    result = check_params(simple_model)
    shape, _ = result["bn1.weight"]
    # Assert
    assert shape == torch.Size([16])


# ---------------------------------------------------------------------------
# Tests: no_grad context
# ---------------------------------------------------------------------------

def test_check_params_within_no_grad_context_reports_learnable(simple_model):
    """check_params inside torch.no_grad() still reports requires_grad correctly."""
    # Arrange
    # Act
    with torch.no_grad():
        result = check_params(simple_model)
    _, status = result["fc1.weight"]
    # Assert
    assert status == "Learnable"


# ---------------------------------------------------------------------------
# Tests: CUDA model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_check_params_on_cuda_model_includes_conv1_weight():
    """check_params on a CUDA model includes 'conv1.weight' key."""
    # Arrange
    model = SimpleModel().cuda()
    # Act
    result = check_params(model)
    # Assert
    assert "conv1.weight" in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_check_params_on_cuda_model_includes_fc1_weight():
    """check_params on a CUDA model includes 'fc1.weight' key."""
    # Arrange
    model = SimpleModel().cuda()
    # Act
    result = check_params(model)
    # Assert
    assert "fc1.weight" in result


# ---------------------------------------------------------------------------
# Tests: consistent ordering
# ---------------------------------------------------------------------------

def test_check_params_produces_consistent_ordering(simple_model):
    """check_params called twice returns keys in the same order."""
    # Arrange
    # Act
    result1 = check_params(simple_model)
    result2 = check_params(simple_model)
    # Assert
    assert list(result1.keys()) == list(result2.keys())


# ---------------------------------------------------------------------------
# Tests: large model performance
# ---------------------------------------------------------------------------

def test_check_params_large_model_param_count_is_200(large_model):
    """check_params on a 100-layer model returns 200 entries (weight + bias)."""
    # Arrange
    # Act
    result = check_params(large_model)
    # Assert
    assert len(result) == 200


def test_check_params_large_model_completes_in_under_one_second(large_model):
    """check_params on a 100-layer model completes in under 1 second."""
    # Arrange
    # Act
    start = time.time()
    result = check_params(large_model)
    elapsed = time.time() - start
    # Assert
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Tests: shared parameters
# ---------------------------------------------------------------------------

def test_check_params_shared_param_model_has_two_entries(shared_param_model):
    """check_params on a shared-param model returns 2 entries (unique weight+bias)."""
    # Arrange
    # Act
    result = check_params(shared_param_model)
    # Assert
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: custom registered parameters
# ---------------------------------------------------------------------------

def test_check_params_includes_directly_assigned_custom_parameter(
    custom_param_model,
):
    """check_params includes directly assigned attribute 'custom_param'."""
    # Arrange
    # Act
    result = check_params(custom_param_model)
    # Assert
    assert "custom_param" in result


def test_check_params_includes_registered_custom_parameter(custom_param_model):
    """check_params includes registered parameter 'named_param'."""
    # Arrange
    # Act
    result = check_params(custom_param_model)
    # Assert
    assert "named_param" in result


def test_check_params_custom_param_shape_is_5_by_5(custom_param_model):
    """check_params reports custom_param shape as torch.Size([5, 5])."""
    # Arrange
    # Act
    result = check_params(custom_param_model)
    # Assert
    assert result["custom_param"][0] == torch.Size([5, 5])


def test_check_params_registered_param_shape_is_3_by_3(custom_param_model):
    """check_params reports named_param shape as torch.Size([3, 3])."""
    # Arrange
    # Act
    result = check_params(custom_param_model)
    # Assert
    assert result["named_param"][0] == torch.Size([3, 3])


# ---------------------------------------------------------------------------
# Tests: buffer exclusion
# ---------------------------------------------------------------------------

def test_check_params_includes_weight_when_model_has_buffer(buffer_model):
    """check_params includes 'fc.weight' even when model has a registered buffer."""
    # Arrange
    # Act
    result = check_params(buffer_model)
    # Assert
    assert "fc.weight" in result


def test_check_params_includes_bias_when_model_has_buffer(buffer_model):
    """check_params includes 'fc.bias' even when model has a registered buffer."""
    # Arrange
    # Act
    result = check_params(buffer_model)
    # Assert
    assert "fc.bias" in result


def test_check_params_excludes_registered_buffer_from_result(buffer_model):
    """check_params excludes registered buffers like 'buffer' from the result."""
    # Arrange
    # Act
    result = check_params(buffer_model)
    # Assert
    assert "buffer" not in result


# ---------------------------------------------------------------------------
# Tests: show defaults to False
# ---------------------------------------------------------------------------

def test_check_params_show_defaults_to_false_no_output(simple_model, capsys):
    """check_params with no show argument prints nothing to stdout."""
    # Arrange
    # Act
    _ = check_params(simple_model)
    captured = capsys.readouterr()
    # Assert
    assert captured.out == ""


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
