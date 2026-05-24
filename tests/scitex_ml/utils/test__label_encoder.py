#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test for scitex_ml.utils._label_encoder

import numpy as np
import pandas as pd
import pytest

from scitex_ml.utils import LabelEncoder

# Try to import torch for testing, but make it optional
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder():
    """Return a fresh LabelEncoder instance."""
    return LabelEncoder()


# ============================================================================
# TestLabelEncoder: initialization
# ============================================================================

def test_init_creates_classes_attribute(encoder):
    """Test that __init__ creates a classes_ attribute on the instance."""
    # Arrange
    # Act
    # Assert
    assert hasattr(encoder, "classes_")


def test_init_classes_array_is_empty(encoder):
    """Test that freshly initialized classes_ array has length zero."""
    # Arrange
    # Act
    # Assert
    assert len(encoder.classes_) == 0


def test_init_classes_attribute_is_numpy_ndarray(encoder):
    """Test that classes_ is stored as a numpy ndarray."""
    # Arrange
    # Act
    # Assert
    assert isinstance(encoder.classes_, np.ndarray)


# ============================================================================
# TestLabelEncoder: fit
# ============================================================================

def test_fit_with_string_labels_returns_self(encoder):
    """Test that fit() returns the encoder instance (fluent interface)."""
    # Arrange
    labels = ["apple", "banana", "cherry"]

    # Act
    result = encoder.fit(labels)

    # Assert
    assert result is encoder


def test_fit_with_string_labels_stores_unique_sorted_classes(encoder):
    """Test that fit() stores unique classes in sorted order."""
    # Arrange
    labels = ["apple", "banana", "cherry"]
    expected_classes = np.array(["apple", "banana", "cherry"])

    # Act
    encoder.fit(labels)

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


def test_fit_initial_stores_provided_classes(encoder):
    """Test that the first fit call stores exactly the provided classes."""
    # Arrange
    expected_first = np.array(["apple", "banana"])

    # Act
    encoder.fit(["apple", "banana"])

    # Assert
    assert np.array_equal(encoder.classes_, expected_first)


def test_fit_incremental_merges_new_classes_with_existing(encoder):
    """Test that a second fit call adds new classes to the existing set."""
    # Arrange
    encoder.fit(["apple", "banana"])
    expected_second = np.array(["apple", "banana", "cherry", "date"])

    # Act
    encoder.fit(["cherry", "date"])

    # Assert
    assert np.array_equal(encoder.classes_, expected_second)


def test_fit_with_duplicates_stores_only_unique_labels(encoder):
    """Test that fitting with duplicate labels deduplicates them."""
    # Arrange
    labels = ["apple", "banana", "apple", "cherry", "banana"]
    expected_classes = np.array(["apple", "banana", "cherry"])

    # Act
    encoder.fit(labels)

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


def test_fit_incremental_with_existing_labels_merges_correctly(encoder):
    """Test incremental fit when some labels already exist in classes."""
    # Arrange
    encoder.fit(["apple", "banana"])
    expected_classes = np.array(["apple", "banana", "cherry", "date"])

    # Act
    encoder.fit(["banana", "cherry", "apple", "date"])

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


# ============================================================================
# TestLabelEncoder: transform
# ============================================================================

def test_transform_after_fit_returns_numeric_encoding(encoder):
    """Test transform() returns integer indices after fitting."""
    # Arrange
    labels = ["apple", "banana", "cherry"]
    encoder.fit(labels)
    expected_encoded = np.array([0, 1, 2])

    # Act
    encoded = encoder.transform(labels)

    # Assert
    assert np.array_equal(encoded, expected_encoded)


def test_transform_with_subset_returns_correct_indices(encoder):
    """Test transforming a subset of fitted labels gives correct indices."""
    # Arrange
    encoder.fit(["apple", "banana", "cherry", "date"])
    subset_labels = ["banana", "cherry"]
    expected_encoded = np.array([1, 2])

    # Act
    encoded = encoder.transform(subset_labels)

    # Assert
    assert np.array_equal(encoded, expected_encoded)


def test_transform_with_unknown_label_raises_value_error(encoder):
    """Test that transform() raises ValueError for unseen labels."""
    # Arrange
    encoder.fit(["apple", "banana"])

    # Act
    ctx = pytest.raises(ValueError, match="y contains new labels")
    # Assert
    with ctx:
        encoder.transform(["apple", "unknown"])


def test_transform_with_empty_classes_raises_value_error(encoder):
    """Test that transform() raises ValueError before any fit call."""
    # Arrange
    # Act
    ctx = pytest.raises(ValueError)
    # Assert
    with ctx:
        encoder.transform(["apple"])


# ============================================================================
# TestLabelEncoder: inverse_transform
# ============================================================================

def test_inverse_transform_recovers_original_labels(encoder):
    """Test inverse_transform() returns the original labels from indices."""
    # Arrange
    labels = ["apple", "banana", "cherry"]
    encoder.fit(labels)
    encoded = encoder.transform(labels)
    expected_decoded = np.array(["apple", "banana", "cherry"])

    # Act
    decoded = encoder.inverse_transform(encoded)

    # Assert
    assert np.array_equal(decoded, expected_decoded)


def test_fit_transform_inverse_roundtrip_preserves_labels(encoder):
    """Test fit -> transform -> inverse_transform produces original labels."""
    # Arrange
    original_labels = ["cat", "dog", "bird", "cat", "dog"]
    expected_decoded = np.array(["cat", "dog", "bird", "cat", "dog"])

    # Act
    encoder.fit(original_labels)
    encoded = encoder.transform(original_labels)
    decoded = encoder.inverse_transform(encoded)

    # Assert
    assert np.array_equal(decoded, expected_decoded)


# ============================================================================
# TestLabelEncoder: numeric / mixed labels
# ============================================================================

def test_fit_with_numeric_labels_sorts_classes_numerically(encoder):
    """Test fitting with numeric labels produces sorted class array."""
    # Arrange
    numeric_labels = [1, 2, 3, 1, 2]
    expected_classes = np.array([1, 2, 3])

    # Act
    encoder.fit(numeric_labels)

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


def test_transform_with_numeric_labels_encodes_by_sorted_order(encoder):
    """Test transform() on numeric labels uses sorted index mapping."""
    # Arrange
    encoder.fit([1, 2, 3, 1, 2])
    expected_encoded = np.array([1, 2, 0])

    # Act
    encoded = encoder.transform([2, 3, 1])

    # Assert
    assert np.array_equal(encoded, expected_encoded)


def test_fit_with_mixed_types_accepts_four_classes(encoder):
    """Test fitting with mixed string/int labels produces correct count."""
    # Arrange
    mixed_labels = ["apple", 1, "banana", 2]

    # Act
    encoder.fit(mixed_labels)

    # Assert
    assert len(encoder.classes_) == 4


def test_transform_with_mixed_types_encodes_subset(encoder):
    """Test transform() on a subset of mixed-type labels returns indices."""
    # Arrange
    encoder.fit(["apple", 1, "banana", 2])

    # Act
    encoded = encoder.transform(["apple", 1])

    # Assert
    assert len(encoded) == 2


# ============================================================================
# TestLabelEncoder: _check_input
# ============================================================================

def test_check_input_with_list_returns_numpy_ndarray(encoder):
    """Test _check_input converts a plain list to a numpy ndarray."""
    # Arrange
    input_list = ["a", "b", "c"]

    # Act
    result = encoder._check_input(input_list)

    # Assert
    assert isinstance(result, np.ndarray)


def test_check_input_with_list_preserves_element_values(encoder):
    """Test _check_input preserves all element values from a list."""
    # Arrange
    input_list = ["a", "b", "c"]

    # Act
    result = encoder._check_input(input_list)

    # Assert
    assert np.array_equal(result, np.array(["a", "b", "c"]))


def test_check_input_with_tuple_returns_numpy_ndarray(encoder):
    """Test _check_input converts a tuple to a numpy ndarray."""
    # Arrange
    input_tuple = ("a", "b", "c")

    # Act
    result = encoder._check_input(input_tuple)

    # Assert
    assert isinstance(result, np.ndarray)


def test_check_input_with_tuple_preserves_element_values(encoder):
    """Test _check_input preserves all element values from a tuple."""
    # Arrange
    input_tuple = ("a", "b", "c")

    # Act
    result = encoder._check_input(input_tuple)

    # Assert
    assert np.array_equal(result, np.array(["a", "b", "c"]))


def test_check_input_with_numpy_array_returns_ndarray_type(encoder):
    """Test _check_input passes through a numpy array unchanged in type."""
    # Arrange
    input_array = np.array(["a", "b", "c"])

    # Act
    result = encoder._check_input(input_array)

    # Assert
    assert isinstance(result, np.ndarray)


def test_check_input_with_numpy_array_preserves_input_values(encoder):
    """Test _check_input preserves all values from a numpy array input."""
    # Arrange
    input_array = np.array(["a", "b", "c"])

    # Act
    result = encoder._check_input(input_array)

    # Assert
    assert np.array_equal(result, input_array)


def test_check_input_with_pandas_series_returns_ndarray_type(encoder):
    """Test _check_input converts a pandas Series to a numpy ndarray."""
    # Arrange
    input_series = pd.Series(["a", "b", "c"])

    # Act
    result = encoder._check_input(input_series)

    # Assert
    assert isinstance(result, np.ndarray)


def test_check_input_with_pandas_series_preserves_values(encoder):
    """Test _check_input preserves element values from a pandas Series."""
    # Arrange
    input_series = pd.Series(["a", "b", "c"])

    # Act
    result = encoder._check_input(input_series)

    # Assert
    assert np.array_equal(result, np.array(["a", "b", "c"]))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_check_input_with_torch_tensor_returns_ndarray_type(encoder):
    """Test _check_input converts a torch tensor to a numpy ndarray."""
    # Arrange
    input_tensor = torch.tensor([1, 2, 3])

    # Act
    result = encoder._check_input(input_tensor)

    # Assert
    assert isinstance(result, np.ndarray)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_check_input_with_torch_tensor_preserves_values(encoder):
    """Test _check_input preserves element values from a torch tensor."""
    # Arrange
    input_tensor = torch.tensor([1, 2, 3])

    # Act
    result = encoder._check_input(input_tensor)

    # Assert
    assert np.array_equal(result, np.array([1, 2, 3]))


# ============================================================================
# TestLabelEncoder: torch tensor workflow
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_fit_with_torch_tensor_stores_correct_classes(encoder):
    """Test fit() on a torch tensor stores the expected unique classes."""
    # Arrange
    tensor_labels = torch.tensor([0, 1, 2, 0, 1])
    expected_classes = np.array([0, 1, 2])

    # Act
    encoder.fit(tensor_labels)

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_transform_with_torch_tensor_encodes_all_values(encoder):
    """Test transform() on a torch tensor returns correct integer encoding."""
    # Arrange
    tensor_labels = torch.tensor([0, 1, 2, 0, 1])
    encoder.fit(tensor_labels)
    expected_encoded = np.array([0, 1, 2, 0, 1])

    # Act
    encoded = encoder.transform(tensor_labels)

    # Assert
    assert np.array_equal(encoded, expected_encoded)


# ============================================================================
# TestLabelEncoder: DataFrame / edge inputs
# ============================================================================

def test_fit_with_dataframe_column_stores_sorted_classes(encoder):
    """Test fit() on a DataFrame Series column sorts classes alphabetically."""
    # Arrange
    df = pd.DataFrame({"labels": ["cat", "dog", "bird"]})
    expected_classes = np.array(["bird", "cat", "dog"])

    # Act
    encoder.fit(df["labels"])

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


def test_fit_with_empty_input_returns_empty_classes(encoder):
    """Test fit() on an empty list keeps classes_ as an empty array."""
    # Arrange
    empty_list: list = []

    # Act
    encoder.fit(empty_list)

    # Assert
    assert len(encoder.classes_) == 0


def test_fit_with_single_label_stores_single_class(encoder):
    """Test fit() with a single label stores just that class."""
    # Arrange
    single_label = ["apple"]
    expected_classes = np.array(["apple"])

    # Act
    encoder.fit(single_label)

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


def test_transform_with_single_label_encodes_to_zero(encoder):
    """Test transform() with a single class always encodes to 0."""
    # Arrange
    encoder.fit(["apple"])
    expected_encoded = np.array([0])

    # Act
    encoded = encoder.transform(["apple"])

    # Assert
    assert np.array_equal(encoded, expected_encoded)


def test_multiple_incremental_fits_accumulates_all_classes(encoder):
    """Test that multiple consecutive fit() calls accumulate every class."""
    # Arrange
    expected_classes = np.array(["a", "b", "c", "d", "e"])

    # Act
    encoder.fit(["a"])
    encoder.fit(["b", "c"])
    encoder.fit(["d"])
    encoder.fit(["a", "e"])  # Include existing label

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


def test_class_ordering_is_consistent_regardless_of_fit_order():
    """Test that two encoders fitted with same classes in different order agree."""
    # Arrange
    labels1 = ["zebra", "apple", "banana"]
    labels2 = ["banana", "zebra", "apple"]
    encoder1 = LabelEncoder()
    encoder2 = LabelEncoder()

    # Act
    encoder1.fit(labels1)
    encoder2.fit(labels2)

    # Assert
    assert np.array_equal(encoder1.classes_, encoder2.classes_)


def test_transform_results_are_independent_of_fit_order():
    """Test transform() returns the same result regardless of fit() order."""
    # Arrange
    encoder1 = LabelEncoder()
    encoder2 = LabelEncoder()
    encoder1.fit(["c", "a", "b"])
    encoder2.fit(["a", "b", "c"])
    test_data = ["b", "a", "c"]

    # Act
    result1 = encoder1.transform(test_data)
    result2 = encoder2.transform(test_data)

    # Assert
    assert np.array_equal(result1, result2)


# ============================================================================
# TestEdgeCases
# ============================================================================

def test_fit_with_none_string_mix_raises_type_error(encoder):
    """Test that fitting with a mix of None and strings raises TypeError."""
    # Arrange
    labels_with_none = ["apple", None, "banana", None]

    # Act
    ctx = pytest.raises(TypeError, match="'<' not supported between instances")
    # Assert
    with ctx:
        encoder.fit(labels_with_none)


def test_fit_with_nan_values_produces_at_least_two_classes(encoder):
    """Test that fitting with NaN among strings produces classes for strings."""
    # Arrange
    labels_with_nan = ["apple", np.nan, "banana"]

    # Act
    encoder.fit(labels_with_nan)

    # Assert
    assert len(encoder.classes_) >= 2


def test_transform_inverse_with_very_long_labels_preserves_content(encoder):
    """Test roundtrip with very long string labels preserves the original text."""
    # Arrange
    long_label = "a" * 1000
    labels = ["short", long_label, "medium_length_label"]
    encoder.fit(labels)

    # Act
    encoded = encoder.transform([long_label])
    decoded = encoder.inverse_transform(encoded)

    # Assert
    assert decoded[0] == long_label


def test_transform_inverse_with_unicode_preserves_emoji(encoder):
    """Test that unicode/emoji labels survive encode-decode roundtrip."""
    # Arrange
    unicode_labels = ["🍎", "🍌", "🍒", "apple"]
    encoder.fit(unicode_labels)
    expected_decoded = np.array(["🍎", "apple"])

    # Act
    encoded = encoder.transform(["🍎", "apple"])
    decoded = encoder.inverse_transform(encoded)

    # Assert
    assert np.array_equal(decoded, expected_decoded)


def test_fit_with_large_labels_creates_correct_class_count(encoder):
    """Test fitting with 1000 distinct labels stores exactly 1000 classes."""
    # Arrange
    large_labels = [f"class_{i}" for i in range(1000)]

    # Act
    encoder.fit(large_labels)

    # Assert
    assert len(encoder.classes_) == 1000


def test_transform_inverse_with_large_subset_preserves_labels(encoder):
    """Test roundtrip on subset of a large label set preserves ordering."""
    # Arrange
    large_labels = [f"class_{i}" for i in range(1000)]
    encoder.fit(large_labels)
    subset = [f"class_{i}" for i in [0, 500, 999]]

    # Act
    encoded = encoder.transform(subset)
    decoded = encoder.inverse_transform(encoded)

    # Assert
    assert np.array_equal(decoded, np.array(subset))


def test_transform_inverse_with_special_characters_preserves_symbols(encoder):
    """Test that labels with special characters survive roundtrip unchanged."""
    # Arrange
    special_labels = [
        "normal",
        "with space",
        "with-dash",
        "with_underscore",
        "with.dot",
        "with@symbol",
    ]
    encoder.fit(special_labels)

    # Act
    encoded = encoder.transform(special_labels)
    decoded = encoder.inverse_transform(encoded)

    # Assert
    assert np.array_equal(decoded, np.array(special_labels))


def test_fit_with_numeric_strings_includes_string_one(encoder):
    """Test that numeric string labels are stored as strings, not coerced."""
    # Arrange
    numeric_strings = ["1", "2", "10", "20"]

    # Act
    encoder.fit(numeric_strings)

    # Assert
    assert "1" in encoder.classes_


def test_fit_with_numeric_strings_orders_by_string_alphabet(encoder):
    """Test that numeric strings are ordered alphabetically, not numerically."""
    # Arrange
    numeric_strings = ["1", "2", "10", "20"]
    expected_order = ["1", "10", "2", "20"]

    # Act
    encoder.fit(numeric_strings)

    # Assert
    assert np.array_equal(encoder.classes_, expected_order)


# ============================================================================
# TestCompatibility
# ============================================================================

def test_transform_matches_sklearn_label_encoder_encoded_output():
    """Test transform() returns same encoded array as sklearn LabelEncoder."""
    # Arrange
    from sklearn.preprocessing import LabelEncoder as SklearnEncoder

    labels = ["apple", "banana", "cherry"]
    our_encoder = LabelEncoder()
    sklearn_encoder = SklearnEncoder()

    # Act
    our_encoder.fit(labels)
    our_encoded = our_encoder.transform(labels)
    sklearn_encoded = sklearn_encoder.fit_transform(labels)

    # Assert
    assert np.array_equal(our_encoded, sklearn_encoded)


def test_inverse_transform_matches_sklearn_label_encoder_output():
    """Test inverse_transform() returns same labels as sklearn LabelEncoder."""
    # Arrange
    from sklearn.preprocessing import LabelEncoder as SklearnEncoder

    labels = ["apple", "banana", "cherry"]
    our_encoder = LabelEncoder()
    sklearn_encoder = SklearnEncoder()

    # Act
    our_encoder.fit(labels)
    our_encoded = our_encoder.transform(labels)
    our_decoded = our_encoder.inverse_transform(our_encoded)
    sklearn_encoded = sklearn_encoder.fit_transform(labels)
    sklearn_decoded = sklearn_encoder.inverse_transform(sklearn_encoded)

    # Assert
    assert np.array_equal(our_decoded, sklearn_decoded)


def test_encoder_is_instance_of_sklearn_label_encoder():
    """Test that our LabelEncoder inherits from sklearn's LabelEncoder."""
    # Arrange
    from sklearn.preprocessing import LabelEncoder as SklearnEncoder

    encoder = LabelEncoder()

    # Act
    result = isinstance(encoder, SklearnEncoder)
    # Assert
    assert result


@pytest.mark.parametrize("method_name", ["fit", "transform", "inverse_transform", "fit_transform"])
def test_encoder_has_sklearn_inherited_method(method_name):
    """Test that the encoder exposes a specific inherited sklearn method."""
    # Arrange
    from sklearn.preprocessing import LabelEncoder as SklearnEncoder

    encoder = LabelEncoder()

    # Act
    result = hasattr(encoder, method_name)
    # Assert
    assert result


def test_fit_transform_method_sets_correct_classes(encoder):
    """Test that the inherited fit_transform() stores the expected classes."""
    # Arrange
    labels = ["apple", "banana", "cherry", "apple"]
    expected_classes = np.array(["apple", "banana", "cherry"])

    # Act
    encoder.fit_transform(labels)

    # Assert
    assert np.array_equal(encoder.classes_, expected_classes)


def test_fit_transform_method_encodes_correctly(encoder):
    """Test that fit_transform() returns correct integer encoding."""
    # Arrange
    labels = ["apple", "banana", "cherry", "apple"]
    expected_encoded = np.array([0, 1, 2, 0])

    # Act
    encoded = encoder.fit_transform(labels)

    # Assert
    assert np.array_equal(encoded, expected_encoded)


# ---------------------------------------------------------------------------
# Run entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
