"""Tests for scitex_ml.sk._to_sktime module."""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
import pandas as pd

from scitex_ml.sk import to_sktime_df


# ── fixtures ───────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_samples, n_chs, seq_len = 10, 3, 50
    return {
        "numpy": np.random.rand(n_samples, n_chs, seq_len),
        "torch": torch.rand(n_samples, n_chs, seq_len),
        "shape": (n_samples, n_chs, seq_len),
    }


# ── numpy input tests ──────────────────────────────────────────

def test_numpy_input_returns_dataframe(sample_data):
    # Arrange
    X_np = sample_data["numpy"]
    # Act
    result = to_sktime_df(X_np)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_numpy_input_preserves_sample_count(sample_data):
    # Arrange
    X_np = sample_data["numpy"]
    n_samples = sample_data["shape"][0]
    # Act
    result = to_sktime_df(X_np)
    # Assert
    assert result.shape[0] == n_samples


def test_numpy_input_preserves_channel_count(sample_data):
    # Arrange
    X_np = sample_data["numpy"]
    n_chs = sample_data["shape"][1]
    # Act
    result = to_sktime_df(X_np)
    # Assert
    assert result.shape[1] == n_chs


def test_numpy_input_cells_are_pandas_series(sample_data):
    # Arrange
    X_np = sample_data["numpy"]
    # Act
    result = to_sktime_df(X_np)
    # Assert
    assert all(
        isinstance(result.iloc[i, j], pd.Series)
        for i in range(result.shape[0])
        for j in range(result.shape[1])
    )


# ── torch input tests ──────────────────────────────────────────

def test_torch_input_returns_dataframe(sample_data):
    # Arrange
    X_torch = sample_data["torch"]
    # Act
    result = to_sktime_df(X_torch)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_torch_input_preserves_sample_count(sample_data):
    # Arrange
    X_torch = sample_data["torch"]
    n_samples = sample_data["shape"][0]
    # Act
    result = to_sktime_df(X_torch)
    # Assert
    assert result.shape[0] == n_samples


def test_torch_input_preserves_channel_count(sample_data):
    # Arrange
    X_torch = sample_data["torch"]
    n_chs = sample_data["shape"][1]
    # Act
    result = to_sktime_df(X_torch)
    # Assert
    assert result.shape[1] == n_chs


# ── dataframe passthrough ──────────────────────────────────────

def test_dataframe_input_returns_same_object():
    # Arrange
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    # Act
    result = to_sktime_df(df)
    # Assert
    assert result is df


# ── invalid input tests ────────────────────────────────────────

def test_list_input_raises_value_error():
    # Arrange
    # Act
    ctx = pytest.raises(ValueError, match="Input X must be")
    # Assert
    with ctx:
        to_sktime_df([1, 2, 3])


def test_string_input_raises_value_error():
    # Arrange
    # Act
    ctx = pytest.raises(ValueError, match="Input X must be")
    # Assert
    with ctx:
        to_sktime_df("invalid")


# ── output structure tests ─────────────────────────────────────

def test_output_dataframe_has_correct_shape(sample_data):
    # Arrange
    X = sample_data["numpy"]
    n_samples, n_chs, seq_len = sample_data["shape"]
    # Act
    result = to_sktime_df(X)
    # Assert
    assert result.shape == (n_samples, n_chs)


def test_output_cells_have_correct_series_length(sample_data):
    # Arrange
    X = sample_data["numpy"]
    seq_len = sample_data["shape"][2]
    # Act
    result = to_sktime_df(X)
    # Assert
    assert all(len(result.iloc[i, j]) == seq_len
               for i in range(result.shape[0])
               for j in range(result.shape[1]))


def test_output_dimension_names_follow_dim_pattern(sample_data):
    # Arrange
    X = sample_data["numpy"]
    n_chs = sample_data["shape"][1]
    # Act
    result = to_sktime_df(X)
    # Assert
    for j in range(n_chs):
        assert result.iloc[0, j].name == f"dim_{j}"


# ── dtype tests ────────────────────────────────────────────────

def test_int32_input_converts_series_to_float64():
    # Arrange
    X = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32)
    # Act
    result = to_sktime_df(X)
    first_series = result.iloc[0, 0]
    # Assert
    assert first_series.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_various_dtypes_output_series_is_float64(dtype):
    # Arrange
    X = np.random.rand(5, 3, 20).astype(dtype)
    # Act
    result = to_sktime_df(X)
    first_series = result.iloc[0, 0]
    # Assert
    assert first_series.dtype == np.float64


# ── edge case tests ────────────────────────────────────────────

def test_empty_input_with_channel_dimension_returns_empty_dataframe():
    # Arrange
    X = np.array([]).reshape(0, 3, 50)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert result.shape[0] == 0


def test_single_sample_returns_one_row_dataframe():
    # Arrange
    X = np.random.rand(1, 5, 100)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert result.shape == (1, 5)


def test_single_sample_cell_has_correct_length():
    # Arrange
    X = np.random.rand(1, 5, 100)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert len(result.iloc[0, 0]) == 100


def test_large_dataset_returns_correct_shape():
    # Arrange
    X = np.random.rand(100, 64, 1000)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert result.shape == (100, 64)


def test_large_dataset_first_cell_has_correct_length():
    # Arrange
    X = np.random.rand(100, 64, 1000)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert len(result.iloc[0, 0]) == 1000


# ── NaN handling ───────────────────────────────────────────────

def test_nan_values_preserved_in_first_five_positions():
    # Arrange
    X = np.random.rand(5, 3, 20)
    X[0, 0, :5] = np.nan
    # Act
    result = to_sktime_df(X)
    first_dim_series = result.iloc[0, 0]
    # Assert
    assert np.isnan(first_dim_series.iloc[:5]).all()


def test_non_nan_values_unchanged_after_position_five():
    # Arrange
    X = np.random.rand(5, 3, 20)
    X[0, 0, :5] = np.nan
    # Act
    result = to_sktime_df(X)
    first_dim_series = result.iloc[0, 0]
    # Assert
    assert not np.isnan(first_dim_series.iloc[5:]).any()
