#!/usr/bin/env python3
# Time-stamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: ./tests/scitex_ml/utils/test__under_sample.py

"""Tests for scitex_ml.utils._under_sample module."""

from collections import Counter

import numpy as np
import pytest

from scitex_ml.utils import under_sample


class TestUnderSample:
    """Test suite for under_sample function."""

    def test_basic_undersampling_returns_correct_total(self):
        """Test that imbalanced classes are downsampled to minority size in total."""
        # Arrange
        y = np.array(["a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "c", "c"])

        # Act
        indices = under_sample(y)

        # Assert
        assert len(indices) == 6

    def test_basic_undersampling_is_balanced_across_classes(self):
        """Test that each class is sampled the same number of times."""
        # Arrange
        y = np.array(["a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "c", "c"])

        # Act
        indices = under_sample(y)
        counts = Counter(y[indices])

        # Assert
        assert counts["a"] == 2 and counts["b"] == 2 and counts["c"] == 2

    def test_numeric_labels_returns_correct_total(self):
        """Test that numeric labels produce the expected total sample count."""
        # Arrange
        y = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])

        # Act
        indices = under_sample(y)

        # Assert
        assert len(indices) == 6

    def test_numeric_labels_returns_balanced_counts(self):
        """Test that numeric-label undersampling produces balanced counts per class."""
        # Arrange
        y = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])

        # Act
        indices = under_sample(y)
        counts = Counter(y[indices])

        # Assert
        assert all(c == 2 for c in counts.values())

    def test_already_balanced_returns_all_indices(self):
        """Test that an already balanced input returns every index."""
        # Arrange
        y = np.array(["x", "x", "y", "y", "z", "z"])

        # Act
        indices = under_sample(y)

        # Assert
        assert set(indices) == set(range(6))

    def test_replace_false_returns_unique_indices(self):
        """Test that replace=False returns all-unique indices."""
        # Arrange
        y = np.array([0, 0, 0, 1])

        # Act
        indices = under_sample(y, replace=False)

        # Assert
        assert len(indices) == len(set(indices))

    def test_replace_true_includes_minority_index(self):
        """Test that the sole minority-class index appears in the result."""
        # Arrange
        y = np.array([0, 0, 0, 0, 0, 1])

        # Act
        indices = under_sample(y, replace=True)

        # Assert
        assert 5 in indices

    def test_indices_are_within_range(self):
        """Test that returned indices are within [0, len(y))."""
        # Arrange
        y = np.array(["a", "b", "c", "b", "c", "a", "c"])

        # Act
        indices = under_sample(y)

        # Assert
        assert bool(np.all(indices >= 0) and np.all(indices < len(y)))

    def test_indices_are_integer_dtype(self):
        """Test that returned indices are integer dtype."""
        # Arrange
        y = np.array(["a", "b", "c", "b", "c", "a", "c"])

        # Act
        indices = under_sample(y)

        # Assert
        assert indices.dtype in [np.int32, np.int64]

    def test_repeated_calls_produce_variety(self):
        """Test that repeated calls produce different index sets (randomness)."""
        # Arrange
        y = np.array([0, 0, 0, 0, 1, 1])

        # Act
        indices_sets = [tuple(sorted(under_sample(y))) for _ in range(10)]

        # Assert
        assert len(set(indices_sets)) > 1

    def test_single_class_returns_all_indices(self):
        """Test that a single-class input returns every index."""
        # Arrange
        y = np.array([1, 1, 1, 1])

        # Act
        indices = under_sample(y)

        # Assert
        assert set(indices) == set(range(4))

    def test_extreme_imbalance_returns_one_per_class(self):
        """Test that one-sample minority forces one sample per class."""
        # Arrange
        y = np.array([0] * 100 + [1])

        # Act
        indices = under_sample(y)
        sampled = y[indices]

        # Assert
        assert int(np.sum(sampled == 0)) == 1 and int(np.sum(sampled == 1)) == 1

    def test_three_classes_returns_correct_total(self):
        """Test total sample count equals minority * n_classes for three classes."""
        # Arrange
        y = np.array([0] * 10 + [1] * 5 + [2] * 3)

        # Act
        indices = under_sample(y)

        # Assert
        assert len(indices) == 9

    def test_three_classes_returns_balanced_counts(self):
        """Test per-class counts equal minority count for three classes."""
        # Arrange
        y = np.array([0] * 10 + [1] * 5 + [2] * 3)

        # Act
        indices = under_sample(y)
        counts = Counter(y[indices])

        # Assert
        assert all(c == 3 for c in counts.values())

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    def test_preserves_data_type_in_sampled_array(self, dtype):
        """Test that y[indices] preserves the input dtype."""
        # Arrange
        y = np.array([1, 1, 1, 2, 2], dtype=dtype)

        # Act
        indices = under_sample(y)
        sampled = y[indices]

        # Assert
        assert sampled.dtype == dtype

    def test_list_input_works_with_numpy_array_conversion(self):
        """Test that a list converted via np.array can be undersampled."""
        # Arrange
        y_array = np.array(["a", "b", "c", "b", "c", "a", "c"])

        # Act
        indices = under_sample(y_array)
        counts = Counter(y_array[indices])

        # Assert
        assert all(c == 2 for c in counts.values())

    def test_empty_array_raises_value_error(self):
        """Test that an empty input array raises ValueError."""
        # Arrange
        y = np.array([])

        # Act
        action = lambda: under_sample(y)

        # Assert
        with pytest.raises(ValueError):
            action()

    def test_deterministic_with_seed_produces_identical_results(self):
        """Test that fixing the numpy seed produces identical results."""
        # Arrange
        y = np.array([0, 0, 0, 0, 1, 1])

        # Act
        np.random.seed(42)
        indices1 = under_sample(y)
        np.random.seed(42)
        indices2 = under_sample(y)

        # Assert
        assert list(indices1) == list(indices2)

    def test_replace_false_with_imbalance_returns_correct_total(self):
        """Test that replace=False with imbalance returns minority * n_classes indices."""
        # Arrange
        y = np.array([0] * 5 + [1] * 2 + [2] * 5)

        # Act
        indices = under_sample(y, replace=False)

        # Assert
        assert len(indices) == 6

    def test_replace_false_with_imbalance_returns_balanced_counts(self):
        """Test that replace=False yields exactly minority-size samples per class."""
        # Arrange
        y = np.array([0] * 5 + [1] * 2 + [2] * 5)

        # Act
        indices = under_sample(y, replace=False)
        counts = Counter(y[indices])

        # Assert
        assert counts[0] == 2 and counts[1] == 2 and counts[2] == 2

    def test_replace_false_with_imbalance_returns_unique_indices(self):
        """Test that replace=False on imbalanced data returns all-unique indices."""
        # Arrange
        y = np.array([0] * 5 + [1] * 2 + [2] * 5)

        # Act
        indices = under_sample(y, replace=False)

        # Assert
        assert len(indices) == len(set(indices))


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_under_sample.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
#
#
# from collections import Counter
#
# import numpy as np
#
#
# def under_sample(y, replace=False):
#     """
#     Input:
#         Labels
#     Return:
#         Indices
#
#     Example:
#         t = ['a', 'b', 'c', 'b', 'c', 'a', 'c']
#         print(under_sample(t))
#         # [5 0 1 3 4 6]
#         print(under_sample(t))
#         # [5 0 1 3 6 2]
#     """
#
#     # find the minority and majority classes
#     class_counts = Counter(y)
#     # majority_class = max(class_counts, key=class_counts.get)
#     minority_class = min(class_counts, key=class_counts.get)
#
#     # compute the number of sample to draw from the majority class using
#     # a negative binomial distribution
#     n_minority_class = class_counts[minority_class]
#     n_majority_resampled = n_minority_class
#
#     # draw randomly with or without replacement
#     indices = np.hstack(
#         [
#             np.random.choice(
#                 np.flatnonzero(y == k),
#                 size=n_majority_resampled,
#                 replace=replace,
#             )
#             for k in class_counts.keys()
#         ]
#     )
#
#     return indices
#
#
# if __name__ == "__main__":
#     t = np.array(["a", "b", "c", "b", "c", "a", "c"])
#     print(under_sample(t))

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_under_sample.py
# --------------------------------------------------------------------------------
