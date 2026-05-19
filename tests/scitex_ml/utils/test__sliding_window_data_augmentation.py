#!/usr/bin/env python3
# Time-stamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: ./tests/scitex_ml/utils/test__sliding_window_data_augmentation.py

"""Tests for scitex_ml.utils._sliding_window_data_augmentation module.

This module provides sliding window data augmentation for time series data,
commonly used in machine learning for creating training samples from continuous
signals.
"""

import random

import numpy as np
import pytest

# Conditionally import torch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scitex_ml.utils import sliding_window_data_augmentation


class TestSlidingWindowDataAugmentationBasic:
    """Basic functionality tests for sliding_window_data_augmentation."""

    def test_basic_1d_array_returns_window_shape(self):
        """Test that 1D input returns a window of the requested size."""
        # Arrange
        x = np.arange(100)
        window_size = 30

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape == (window_size,)

    def test_basic_1d_array_values_are_subset(self):
        """Test that returned 1D window values are taken from the input."""
        # Arrange
        x = np.arange(100)
        window_size = 30

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert bool(np.all(np.isin(result, x)))

    def test_basic_1d_array_values_are_consecutive(self):
        """Test that the 1D window holds consecutive integers."""
        # Arrange
        x = np.arange(100)
        window_size = 30

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert bool(np.all(np.diff(result) == 1))

    def test_2d_array_returns_window_shape(self):
        """Test that 2D input returns (channels, window_size)."""
        # Arrange
        x = np.random.rand(5, 1000)
        window_size = 200

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape == (5, window_size)

    def test_2d_array_preserves_channel_count(self):
        """Test that 2D input preserves the channel dimension."""
        # Arrange
        x = np.random.rand(5, 1000)
        window_size = 200

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape[0] == x.shape[0]

    def test_3d_array_returns_window_shape(self):
        """Test that 3D input returns (batch, channels, window_size)."""
        # Arrange
        x = np.random.rand(10, 5, 500)
        window_size = 100

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape == (10, 5, window_size)

    def test_3d_array_preserves_leading_dims(self):
        """Test that 3D input preserves batch and channel dimensions."""
        # Arrange
        x = np.random.rand(10, 5, 500)
        window_size = 100

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape[:-1] == x.shape[:-1]

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_tensor_input_returns_tensor(self):
        """Test that PyTorch tensor input returns a PyTorch tensor."""
        # Arrange
        x = torch.randn(3, 4, 1000)
        window_size = 256

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert isinstance(result, torch.Tensor)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_tensor_input_returns_window_shape(self):
        """Test that PyTorch tensor input returns the requested window shape."""
        # Arrange
        x = torch.randn(3, 4, 1000)
        window_size = 256

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape == (3, 4, window_size)

    def test_window_size_equals_array_size_returns_full(self):
        """Test that window size equal to array length returns the whole array."""
        # Arrange
        x = np.arange(50)
        window_size = 50

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert list(result) == list(x)

    def test_zero_window_size_returns_empty(self):
        """Test that a zero window size returns an empty array."""
        # Arrange
        x = np.arange(100)
        window_size = 0

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape == (0,)

    def test_float_array_preserves_dtype(self):
        """Test that float input preserves its dtype."""
        # Arrange
        x = np.random.randn(200).astype(np.float32)
        window_size = 50

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.dtype == np.float32

    def test_float_array_returns_window_shape(self):
        """Test that float input returns the requested window shape."""
        # Arrange
        x = np.random.randn(200).astype(np.float32)
        window_size = 50

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape == (window_size,)

    def test_preserves_data_continuity_in_sine_window(self):
        """Test that consecutive samples in a smooth signal remain close."""
        # Arrange
        x = np.sin(np.linspace(0, 10 * np.pi, 1000))
        window_size = 200

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert bool(np.all(np.abs(np.diff(result)) < 0.1))

    @pytest.mark.parametrize(
        "shape,window_size",
        [
            ((100,), 50),
            ((10, 100), 30),
            ((5, 10, 200), 100),
            ((2, 3, 4, 500), 250),
        ],
    )
    def test_various_shapes_last_dim_equals_window(self, shape, window_size):
        """Test that the last dim of the result equals the window size."""
        # Arrange
        x = np.random.rand(*shape)

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape[-1] == window_size

    @pytest.mark.parametrize(
        "shape,window_size",
        [
            ((100,), 50),
            ((10, 100), 30),
            ((5, 10, 200), 100),
            ((2, 3, 4, 500), 250),
        ],
    )
    def test_various_shapes_leading_dims_preserved(self, shape, window_size):
        """Test that all leading dims of the result match the input."""
        # Arrange
        x = np.random.rand(*shape)

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert result.shape[:-1] == x.shape[:-1]

    def test_window_inside_array_bounds(self):
        """Test that the returned window stays within the input array."""
        # Arrange
        x = np.arange(100)
        window_size = 20

        # Act
        result = sliding_window_data_augmentation(x, window_size)

        # Assert
        assert bool(np.all(np.isin(result, x))) and len(result) == window_size


class TestSlidingWindowDataAugmentationIntegration:
    """Integration tests with ML workflows."""

    def test_training_data_generation_collects_expected_shape(self):
        """Test that stacking augmented samples yields the expected tensor shape."""
        # Arrange
        n_samples = 10000
        n_channels = 32
        data = np.random.randn(n_channels, n_samples)
        window_size = 256
        n_augmentations = 100

        # Act
        training_samples = [
            sliding_window_data_augmentation(data, window_size)
            for _ in range(n_augmentations)
        ]
        training_data = np.array(training_samples)

        # Assert
        assert training_data.shape == (n_augmentations, n_channels, window_size)

    def test_training_data_generation_yields_diverse_windows(self):
        """Test that augmentation produces windows starting at different positions."""
        # Arrange
        n_samples = 10000
        n_channels = 32
        data = np.random.randn(n_channels, n_samples)
        window_size = 256
        n_augmentations = 100

        # Act
        first_values = [
            sliding_window_data_augmentation(data, window_size)[0, 0]
            for _ in range(n_augmentations)
        ]

        # Assert
        assert len(set(first_values)) > 1

    def test_multi_scale_augmentation_preserves_shapes(self):
        """Test that augmentation at multiple window sizes preserves expected shape."""
        # Arrange
        x = np.random.randn(8, 10000)
        window_sizes = [100, 200, 500, 1000]

        # Act
        multi_scale_samples = {
            size: np.array(
                [sliding_window_data_augmentation(x, size) for _ in range(10)]
            )
            for size in window_sizes
        }

        # Assert
        assert all(
            multi_scale_samples[size].shape == (10, 8, size) for size in window_sizes
        )

    def test_augmentation_with_labels_yields_expected_data_shape(self):
        """Test that augmented data has the expected (n, channels, window) shape."""
        # Arrange
        n_samples = 10000
        n_channels = 16
        data = np.random.randn(n_channels, n_samples)
        window_size = 200
        n_aug = 50
        augmented_data = []
        for _ in range(n_aug):
            start = random.randint(0, n_samples - window_size)
            augmented_data.append(data[:, start : start + window_size])

        # Act
        augmented_array = np.array(augmented_data)

        # Assert
        assert augmented_array.shape == (n_aug, n_channels, window_size)

    def test_augmentation_with_labels_yields_majority_label_array(self):
        """Test that majority-label aggregation produces the expected shape."""
        # Arrange
        n_samples = 10000
        labels = np.repeat([0, 1, 2, 3], n_samples // 4)
        window_size = 200
        n_aug = 50

        # Act
        augmented_labels = []
        for _ in range(n_aug):
            start = random.randint(0, n_samples - window_size)
            window_labels = labels[start : start + window_size]
            unique, counts = np.unique(window_labels, return_counts=True)
            augmented_labels.append(unique[np.argmax(counts)])
        augmented_labels = np.array(augmented_labels)

        # Assert
        assert augmented_labels.shape == (n_aug,) and bool(
            np.all(np.isin(augmented_labels, [0, 1, 2, 3]))
        )


class TestSlidingWindowDataAugmentationDocumentation:
    """Test documentation and usage examples."""

    def test_function_signature_matches_spec(self):
        """Test that the function signature matches the documented one."""
        # Arrange
        import inspect

        sig = inspect.signature(sliding_window_data_augmentation)

        # Act
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["x", "window_size_pts"]

    def test_example_eeg_processing_window_shape(self):
        """Test EEG-like augmentation returns (channels, window_samples)."""
        # Arrange
        fs = 256
        duration = 10
        n_channels = 32
        n_samples = fs * duration
        eeg = np.random.randn(n_channels, n_samples)
        window_samples = int(1.0 * fs)

        # Act
        augmented = sliding_window_data_augmentation(eeg, window_samples)

        # Assert
        assert augmented.shape == (n_channels, window_samples)

    def test_example_audio_processing_window_shape(self):
        """Test stereo-audio augmentation returns (channels, window_samples)."""
        # Arrange
        fs = 44100
        duration = 5
        n_channels = 2
        n_samples = fs * duration
        audio = np.random.randn(n_channels, n_samples) * 0.1
        window_samples = int(0.5 * fs)

        # Act
        augmented = sliding_window_data_augmentation(audio, window_samples)

        # Assert
        assert augmented.shape == (n_channels, window_samples)

    def test_example_sensor_data_window_shape(self):
        """Test multi-sensor augmentation returns (sensors, window_size)."""
        # Arrange
        n_sensors = 5
        n_samples = 100000
        sensor_data = np.random.randn(n_sensors, n_samples)

        # Act
        augmented = sliding_window_data_augmentation(sensor_data, 500)

        # Assert
        assert augmented.shape == (n_sensors, 500)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_sliding_window_data_augmentation.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-01-24 13:56:36 (ywatanabe)"
#
# import random
#
#
# def sliding_window_data_augmentation(x, window_size_pts):
#     start = random.randint(0, x.shape[-1] - window_size_pts)
#     end = start + window_size_pts
#     return x[..., start:end]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_sliding_window_data_augmentation.py
# --------------------------------------------------------------------------------
