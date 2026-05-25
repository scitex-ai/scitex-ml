"""Tests for scitex_ml.utils._default_dataset module."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from scitex_ml.utils import DefaultDataset


# ── initialization tests ───────────────────────────────────────

def test_init_single_array_sets_correct_length():
    # Arrange
    X = np.random.rand(100, 10)
    # Act
    ds = DefaultDataset([X])
    # Assert
    assert len(ds) == 100


def test_init_single_array_has_none_transform_by_default():
    # Arrange
    X = np.random.rand(100, 10)
    # Act
    ds = DefaultDataset([X])
    # Assert
    assert ds.transform is None


def test_init_single_array_stores_arrs_list():
    # Arrange
    X = np.random.rand(100, 10)
    # Act
    ds = DefaultDataset([X])
    # Assert
    assert ds.arrs_list == [X]


def test_init_multiple_arrays_sets_correct_length():
    # Arrange
    n = 50
    X = np.random.rand(n, 19, 1000)
    T = np.random.randint(0, 4, size=(n, 1))
    S = np.random.randint(0, 999, size=(n, 1))
    # Act
    ds = DefaultDataset([X, T, S])
    # Assert
    assert len(ds) == n


def test_init_multiple_arrays_stores_three_arrays():
    # Arrange
    n = 50
    X = np.random.rand(n, 19, 1000)
    T = np.random.randint(0, 4, size=(n, 1))
    S = np.random.randint(0, 999, size=(n, 1))
    # Act
    ds = DefaultDataset([X, T, S])
    # Assert
    assert len(ds.arrs_list) == 3


# ── __getitem__ tests ──────────────────────────────────────────

def test_getitem_single_array_first_element_has_one_item():
    # Arrange
    X = np.random.rand(10, 5)
    ds = DefaultDataset([X])
    # Act
    item = ds[0]
    # Assert
    assert len(item) == 1


def test_getitem_single_array_first_element_matches_original():
    # Arrange
    X = np.random.rand(10, 5)
    ds = DefaultDataset([X])
    # Act
    item = ds[0]
    # Assert
    assert np.array_equal(item[0], X[0])


def test_getitem_single_array_last_element_matches_original():
    # Arrange
    X = np.random.rand(10, 5)
    ds = DefaultDataset([X])
    # Act
    item = ds[9]
    # Assert
    assert np.array_equal(item[0], X[9])


def test_getitem_multiple_arrays_item_has_three_elements():
    # Arrange
    n = 20
    X = np.random.rand(n, 10)
    T = np.random.randint(0, 4, size=(n,))
    S = np.random.randint(0, 999, size=(n,))
    ds = DefaultDataset([X, T, S])
    # Act
    item = ds[0]
    # Assert
    assert len(item) == 3


def test_getitem_multiple_arrays_first_element_matches_original():
    # Arrange
    n = 20
    X = np.random.rand(n, 10)
    T = np.random.randint(0, 4, size=(n,))
    S = np.random.randint(0, 999, size=(n,))
    ds = DefaultDataset([X, T, S])
    # Act
    item = ds[0]
    # Assert
    assert np.array_equal(item[0], X[0])


def test_getitem_multiple_arrays_second_element_matches_original():
    # Arrange
    n = 20
    X = np.random.rand(n, 10)
    T = np.random.randint(0, 4, size=(n,))
    S = np.random.randint(0, 999, size=(n,))
    ds = DefaultDataset([X, T, S])
    # Act
    item = ds[0]
    # Assert
    assert item[1] == T[0]


# ── transform tests ────────────────────────────────────────────

def test_transform_applied_to_first_array():
    # Arrange
    def double_transform(x):
        return x * 2
    X = np.ones((10, 5))
    T = np.ones((10,))
    ds = DefaultDataset([X, T], transform=double_transform)
    # Act
    item = ds[0]
    # Assert
    assert np.allclose(item[0], 2.0)


def test_transform_not_applied_to_second_array():
    # Arrange
    def double_transform(x):
        return x * 2
    X = np.ones((10, 5))
    T = np.ones((10,))
    ds = DefaultDataset([X, T], transform=double_transform)
    # Act
    item = ds[0]
    # Assert
    assert item[1] == 1.0


def test_transform_preserves_original_dtype():
    # Arrange
    def add_noise(x):
        return x + np.random.randn(*x.shape) * 0.01
    X = np.ones((10, 5), dtype=np.float32)
    ds = DefaultDataset([X], transform=add_noise)
    # Act
    item = ds[0]
    # Assert
    assert item[0].dtype == np.float32


def test_complex_transform_normalizes_data_correctly():
    # Arrange
    def normalize_transform(x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8)
    X = np.random.rand(20, 10) * 100 + 50
    ds = DefaultDataset([X], transform=normalize_transform)
    # Act
    item = ds[0]
    transformed = normalize_transform(X[0].astype(np.float64)).astype(X.dtype)
    # Assert
    assert np.allclose(item[0], transformed)


# ── error handling tests ───────────────────────────────────────

def test_zero_length_array_raises_assertion_error():
    # Arrange
    X = np.array([])
    # Act
    ctx = pytest.raises(AssertionError)
    # Assert
    with ctx:
        DefaultDataset([X])


def test_empty_arrays_list_raises_index_error():
    # Arrange
    # Act
    ctx = pytest.raises(IndexError)
    # Assert
    with ctx:
        DefaultDataset([])


def test_index_out_of_bounds_positive_raises_index_error():
    # Arrange
    X = np.random.rand(10, 5)
    ds = DefaultDataset([X])
    # Act
    ctx = pytest.raises(IndexError)
    # Assert
    with ctx:
        _ = ds[10]


def test_index_out_of_bounds_negative_raises_index_error():
    # Arrange
    X = np.random.rand(10, 5)
    ds = DefaultDataset([X])
    # Act
    ctx = pytest.raises(IndexError)
    # Assert
    with ctx:
        _ = ds[-11]


# ── dtype tests ────────────────────────────────────────────────

def test_different_dtypes_preserved_for_float32():
    # Arrange
    X = np.random.rand(10, 5).astype(np.float32)
    T = np.random.randint(0, 4, size=(10,)).astype(np.int64)
    S = np.random.rand(10).astype(np.float64)
    ds = DefaultDataset([X, T, S])
    # Act
    item = ds[0]
    # Assert
    assert item[0].dtype == np.float32


def test_different_dtypes_preserved_for_int64():
    # Arrange
    X = np.random.rand(10, 5).astype(np.float32)
    T = np.random.randint(0, 4, size=(10,)).astype(np.int64)
    S = np.random.rand(10).astype(np.float64)
    ds = DefaultDataset([X, T, S])
    # Act
    item = ds[0]
    # Assert
    assert item[1].dtype == np.int64


# ── misc tests ─────────────────────────────────────────────────

def test_mismatched_lengths_uses_first_array_length():
    # Arrange
    X = np.random.rand(10, 5)
    T = np.random.rand(8)
    # Act
    ds = DefaultDataset([X, T])
    # Assert
    assert len(ds) == 10


def test_dataloader_iterates_correct_batch_count():
    # Arrange
    n = 100
    X = np.random.rand(n, 10)
    T = np.random.randint(0, 4, size=(n,))
    ds = DefaultDataset([X, T])
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    # Act
    batch_count = sum(1 for _ in loader)
    # Assert
    assert batch_count == (n + 15) // 16


def test_negative_indexing_last_element_matches_original():
    # Arrange
    X = np.random.rand(10, 5)
    ds = DefaultDataset([X])
    # Act
    last_item = ds[-1]
    # Assert
    assert np.array_equal(last_item[0], X[-1])


def test_multidimensional_arrays_first_item_has_correct_shape():
    # Arrange
    X_3d = np.random.rand(50, 19, 1000)
    X_2d = np.random.rand(50, 100)
    T_1d = np.random.randint(0, 4, size=(50,))
    ds = DefaultDataset([X_3d, X_2d, T_1d])
    # Act
    item = ds[0]
    # Assert
    assert item[0].shape == (19, 1000)


def test_docstring_example_sets_correct_dataset_length():
    # Arrange
    n = 1024
    n_chs = 19
    X = np.random.rand(n, n_chs, 1000)
    T = np.random.randint(0, 4, size=(n, 1))
    S = np.random.randint(0, 999, size=(n, 1))
    Sr = np.random.randint(0, 4, size=(n, 1))
    arrs_list = [X, T, S, Sr]
    # Act
    ds = DefaultDataset(arrs_list)
    # Assert
    assert len(ds) == 1024
