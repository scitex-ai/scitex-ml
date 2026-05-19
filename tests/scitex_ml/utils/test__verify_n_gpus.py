#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: ./tests/scitex_ml/utils/test__verify_n_gpus.py

"""Tests for scitex_ml.utils._verify_n_gpus module.

verify_n_gpus reads the live ``torch.cuda.device_count()`` and warns when the
requested count exceeds the available count. These tests exercise the function
against the real CUDA state (no monkeypatch / mock per ecosystem rules
STX-NM002 and PA-306) — they branch on the actual device count so the
assertions describe genuine production behaviour rather than rewritten
internals.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from scitex_ml.utils import verify_n_gpus


def _actual_gpu_count():
    """Return the real ``torch.cuda.device_count()``."""
    return torch.cuda.device_count()


class TestVerifyNGpusAgainstRealCuda:
    """verify_n_gpus exercised against the live CUDA device count."""

    def test_request_zero_gpus_returns_zero(self):
        """Test that requesting zero GPUs returns zero regardless of availability."""
        # Arrange
        requested = 0

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = verify_n_gpus(requested)

        # Assert
        assert result == 0

    def test_request_available_count_returns_available_count(self):
        """Test that requesting exactly available count returns available count."""
        # Arrange
        available = _actual_gpu_count()
        requested = available

        # Act
        result = verify_n_gpus(requested)

        # Assert
        assert result == available

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 1,
        reason="requires at least 1 CUDA device",
    )
    def test_request_below_available_returns_request(self):
        """Test that requesting fewer than available returns the request."""
        # Arrange
        available = _actual_gpu_count()
        requested = max(0, available - 1)

        # Act
        result = verify_n_gpus(requested)

        # Assert
        assert result == requested

    def test_request_above_available_returns_available(self):
        """Test that exceeding available count clamps the result to available."""
        # Arrange
        available = _actual_gpu_count()
        requested = available + 10

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = verify_n_gpus(requested)

        # Assert
        assert result == available

    def test_request_above_available_emits_user_warning(self):
        """Test that exceeding available count emits a UserWarning."""
        # Arrange
        available = _actual_gpu_count()
        requested = available + 10

        # Act
        action = lambda: verify_n_gpus(requested)

        # Assert
        with pytest.warns(UserWarning):
            action()

    def test_request_above_available_warning_mentions_requested_value(self):
        """Test that the warning message includes the requested count."""
        # Arrange
        available = _actual_gpu_count()
        requested = available + 7

        # Act
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            verify_n_gpus(requested)
        message = str(captured[0].message)

        # Assert
        assert f"N_GPUS ({requested})" in message

    def test_request_above_available_warning_mentions_available_count(self):
        """Test that the warning message includes the available count."""
        # Arrange
        available = _actual_gpu_count()
        requested = available + 7

        # Act
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            verify_n_gpus(requested)
        message = str(captured[0].message)

        # Assert
        assert f"= {available}" in message

    def test_request_above_available_warning_mentions_cuda_visible_devices(self):
        """Test that the warning text references CUDA_VISIBLE_DEVICES."""
        # Arrange
        available = _actual_gpu_count()
        requested = available + 7

        # Act
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            verify_n_gpus(requested)
        message = str(captured[0].message)

        # Assert
        assert "$CUDA_VISIBLE_DEVICES" in message

    def test_request_one_million_returns_available(self):
        """Test that an extreme request is clamped to the available count."""
        # Arrange
        available = _actual_gpu_count()
        requested = 1_000_000

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = verify_n_gpus(requested)

        # Assert
        assert result == available

    def test_negative_request_returned_unchanged(self):
        """Test that a negative request value is returned unchanged."""
        # Arrange
        requested = -1

        # Act
        result = verify_n_gpus(requested)

        # Assert
        assert result == -1

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 1,
        reason="requires at least 1 CUDA device for float test",
    )
    def test_float_request_below_available_returned_unchanged(self):
        """Test that a float request not exceeding available is returned unchanged."""
        # Arrange
        available = _actual_gpu_count()
        requested = float(available) - 0.5

        # Act
        result = verify_n_gpus(requested)

        # Assert
        assert result == requested

    def test_string_input_raises_type_error(self):
        """Test that a string input raises TypeError on comparison."""
        # Arrange
        bad_input = "2"

        # Act
        action = lambda: verify_n_gpus(bad_input)

        # Assert
        with pytest.raises(TypeError):
            action()

    def test_none_input_raises_type_error(self):
        """Test that a None input raises TypeError on comparison."""
        # Arrange
        bad_input = None

        # Act
        action = lambda: verify_n_gpus(bad_input)

        # Assert
        with pytest.raises(TypeError):
            action()

    def test_repeated_calls_return_same_value(self):
        """Test that repeated calls return the same value for the same input."""
        # Arrange
        available = _actual_gpu_count()
        requested = available

        # Act
        first = verify_n_gpus(requested)
        second = verify_n_gpus(requested)

        # Assert
        assert first == second


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_verify_n_gpus.py
# --------------------------------------------------------------------------------
# import torch
# import warnings
#
#
# def verify_n_gpus(n_gpus):
#     if torch.cuda.device_count() < n_gpus:
#         warnings.warn(
#             f"N_GPUS ({n_gpus}) is larger "
#             f"than n_gpus torch can acesses (= {torch.cuda.device_count()})"
#             f"Please check $CUDA_VISIBLE_DEVICES and your setting in this script.",
#             UserWarning,
#         )
#         return torch.cuda.device_count()
#
#     else:
#         return n_gpus

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_verify_n_gpus.py
# --------------------------------------------------------------------------------
