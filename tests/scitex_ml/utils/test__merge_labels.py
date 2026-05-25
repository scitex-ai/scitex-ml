#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2026-05-25 00:00:00 (ywatanabe)"
# File: ./tests/scitex_ml/utils/test__merge_labels.py

"""Tests for scitex_ml.utils._merge_labels module."""

import numpy as np
import pytest

from scitex_ml.utils import merge_labels


# ---------------------------------------------------------------------------
# Tests: basic merging
# ---------------------------------------------------------------------------


def test_merge_two_labels_basic_produces_dash_joined_strings():
    """Test merging two label arrays produces dash-joined strings."""
    # Arrange
    y1 = np.array([0, 1, 0, 1])
    y2 = np.array([0, 0, 1, 1])
    expected = ["0-0", "1-0", "0-1", "1-1"]

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert list(result) == expected


def test_merge_three_labels_produces_dash_joined_strings():
    """Test merging three label arrays produces dash-joined strings."""
    # Arrange
    y1 = np.array([0, 1, 2])
    y2 = np.array([3, 4, 5])
    y3 = np.array([6, 7, 8])
    expected = ["0-3-6", "1-4-7", "2-5-8"]

    # Act
    result = merge_labels(y1, y2, y3)

    # Assert
    assert list(result) == expected


def test_merge_labels_to_int_true_returns_integer_dtype():
    """Test that to_int=True produces integer dtype."""
    # Arrange
    y1 = np.array([0, 1, 0, 1, 0])
    y2 = np.array([0, 0, 1, 1, 0])

    # Act
    result = merge_labels(y1, y2, to_int=True)

    # Assert
    assert result.dtype in [np.int32, np.int64, int]


def test_merge_labels_to_int_true_produces_unique_combinations():
    """Test that to_int=True yields one integer per unique pair."""
    # Arrange
    y1 = np.array([0, 1, 0, 1, 0])
    y2 = np.array([0, 0, 1, 1, 0])

    # Act
    result = merge_labels(y1, y2, to_int=True)

    # Assert
    assert len(np.unique(result)) == 4


def test_single_label_array_returns_input_unchanged():
    """Test that single label array is returned unchanged."""
    # Arrange
    y = np.array([1, 2, 3, 4])

    # Act
    result = merge_labels(y)

    # Assert
    assert list(result) == [1, 2, 3, 4]


def test_empty_arrays_returns_empty_ndarray():
    """Test merging empty arrays returns an empty ndarray."""
    # Arrange
    y1 = np.array([], dtype=int)
    y2 = np.array([], dtype=int)

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert isinstance(result, np.ndarray) and len(result) == 0


def test_mismatched_lengths_truncates_to_shorter_array():
    """Test that mismatched array lengths use zip behavior (truncate to shorter)."""
    # Arrange
    y1 = np.array([0, 1, 2])
    y2 = np.array([0, 1])
    expected = ["0-0", "1-1"]

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert list(result) == expected


def test_string_labels_produces_dash_joined_output():
    """Test merging string labels produces dash-joined strings."""
    # Arrange
    y1 = ["cat", "dog", "cat"]
    y2 = ["A", "B", "A"]
    expected = ["cat-A", "dog-B", "cat-A"]

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert list(result) == expected


def test_string_labels_to_int_collapses_duplicate_pairs():
    """Test that duplicate string-pair labels map to the same integer."""
    # Arrange
    y1 = ["cat", "dog", "cat", "dog"]
    y2 = ["A", "B", "A", "B"]

    # Act
    result = merge_labels(y1, y2, to_int=True)

    # Assert
    assert len(np.unique(result)) == 2


def test_mixed_type_labels_produces_dash_joined_strings():
    """Test merging int+string labels produces dash-joined strings."""
    # Arrange
    y1 = np.array([0, 1, 2])
    y2 = np.array(["A", "B", "C"])
    expected = ["0-A", "1-B", "2-C"]

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert list(result) == expected


def test_many_labels_preserves_sample_count():
    """Test that merging many label arrays preserves sample count."""
    # Arrange
    n_samples = 100
    label_arrays = [np.random.randint(0, 3, size=n_samples) for _ in range(5)]

    # Act
    result = merge_labels(*label_arrays)

    # Assert
    assert len(result) == n_samples


def test_many_labels_each_output_has_one_separator_per_source_array():
    """Test that each merged label has one separator per source array."""
    # Arrange
    n_samples = 100
    n_label_arrays = 5
    label_arrays = [
        np.random.randint(0, 3, size=n_samples) for _ in range(n_label_arrays)
    ]

    # Act
    result = merge_labels(*label_arrays)

    # Assert
    assert all(len(str(label).split("-")) == n_label_arrays for label in result)


def test_to_int_maps_duplicate_combinations_to_same_integer():
    """Test that to_int conversion maps duplicate combinations to same integer."""
    # Arrange
    y1 = np.array([0, 0, 1, 1, 2, 2])
    y2 = np.array([0, 0, 0, 0, 0, 0])

    # Act
    result = merge_labels(y1, y2, to_int=True)

    # Assert
    assert (
        result[0] == result[1] and result[2] == result[3] and result[4] == result[5]
    )


def test_merge_labels_always_returns_numpy_ndarray():
    """Test that output is always numpy array."""
    # Arrange
    y1 = [0, 1, 2]
    y2 = [3, 4, 5]

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert isinstance(result, np.ndarray)


def test_deterministic_int_mapping_is_repeatable_across_calls():
    """Test that integer mapping is deterministic across calls."""
    # Arrange
    y1 = np.array([1, 0, 1, 0])
    y2 = np.array([1, 1, 0, 0])

    # Act
    result1 = merge_labels(y1, y2, to_int=True)
    result2 = merge_labels(y1, y2, to_int=True)

    # Assert
    assert list(result1) == list(result2)


def test_numeric_string_labels_produces_dash_joined_string():
    """Test that numeric values are converted to dash-joined strings."""
    # Arrange
    y1 = np.array([10, 20, 30])
    y2 = np.array([100, 200, 300])
    expected = ["10-100", "20-200", "30-300"]

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert list(result) == expected


def test_float_labels_first_pair_is_dash_joined_correctly():
    """Test that first pair of float labels is dash-joined as expected."""
    # Arrange
    y1 = np.array([0.5, 1.5, 2.5])
    y2 = np.array([1.1, 2.2, 3.3])

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert "0.5-1.1" in result


def test_float_labels_middle_pair_is_dash_joined_correctly():
    """Test that middle pair of float labels is dash-joined as expected."""
    # Arrange
    y1 = np.array([0.5, 1.5, 2.5])
    y2 = np.array([1.1, 2.2, 3.3])

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert "1.5-2.2" in result


def test_float_labels_last_pair_is_dash_joined_correctly():
    """Test that last pair of float labels is dash-joined as expected."""
    # Arrange
    y1 = np.array([0.5, 1.5, 2.5])
    y2 = np.array([1.1, 2.2, 3.3])

    # Act
    result = merge_labels(y1, y2)

    # Assert
    assert "2.5-3.3" in result


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
