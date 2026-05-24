"""Tests for scitex_ml.sklearn.to_sktime module."""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
import pandas as pd

from scitex_ml.sklearn.to_sktime import to_sktime_df


# ── fixtures ───────────────────────────────────────────────────

@pytest.fixture
def sample_numpy_data():
    """Create sample numpy array data."""
    np.random.seed(42)
    return np.random.rand(10, 3, 50)  # 10 samples, 3 channels, 50 time points


@pytest.fixture
def sample_torch_data(sample_numpy_data):
    """Create sample torch tensor data."""
    return torch.from_numpy(sample_numpy_data.copy())


@pytest.fixture
def sample_pandas_data():
    """Create sample pandas DataFrame data in sktime format."""
    data = []
    for _ in range(5):
        row = [pd.Series(np.random.rand(20), name=f"dim_{j}") for j in range(2)]
        data.append(row)
    return pd.DataFrame(data, columns=[0, 1])


# ── numpy input tests ──────────────────────────────────────────

def test_numpy_input_returns_dataframe(sample_numpy_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_numpy_data)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_numpy_input_preserves_sample_count(sample_numpy_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_numpy_data)
    # Assert
    assert len(result) == sample_numpy_data.shape[0]


def test_numpy_input_preserves_channel_count(sample_numpy_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_numpy_data)
    # Assert
    assert len(result.columns) == sample_numpy_data.shape[1]


def test_numpy_input_first_cell_is_pandas_series(sample_numpy_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_numpy_data)
    first_channel = result.iloc[0, 0]
    # Assert
    assert isinstance(first_channel, pd.Series)


def test_numpy_input_first_cell_has_correct_sequence_length(sample_numpy_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_numpy_data)
    first_channel = result.iloc[0, 0]
    # Assert
    assert len(first_channel) == sample_numpy_data.shape[2]


def test_numpy_input_first_cell_name_is_dim_0(sample_numpy_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_numpy_data)
    first_channel = result.iloc[0, 0]
    # Assert
    assert first_channel.name == "dim_0"


# ── torch input tests ──────────────────────────────────────────

def test_torch_input_returns_dataframe(sample_torch_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_torch_data)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_torch_input_preserves_sample_count(sample_torch_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_torch_data)
    # Assert
    assert len(result) == sample_torch_data.shape[0]


def test_torch_input_preserves_channel_count(sample_torch_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_torch_data)
    # Assert
    assert len(result.columns) == sample_torch_data.shape[1]


def test_torch_input_first_cell_is_pandas_series(sample_torch_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_torch_data)
    first_channel = result.iloc[0, 0]
    # Assert
    assert isinstance(first_channel, pd.Series)


def test_torch_input_first_cell_has_correct_sequence_length(sample_torch_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_torch_data)
    first_channel = result.iloc[0, 0]
    # Assert
    assert len(first_channel) == sample_torch_data.shape[2]


# ── pandas passthrough ─────────────────────────────────────────

def test_pandas_input_returns_same_object(sample_pandas_data):
    # Arrange
    # Act
    result = to_sktime_df(sample_pandas_data)
    # Assert
    assert result is sample_pandas_data


# ── dtype conversion ───────────────────────────────────────────

def test_int32_input_converts_cell_dtype_to_float64():
    # Arrange
    X = np.random.randint(0, 10, size=(5, 2, 10)).astype(np.int32)
    # Act
    result = to_sktime_df(X)
    first_channel = result.iloc[0, 0]
    # Assert
    assert first_channel.dtype == np.float64


# ── dimension naming tests ─────────────────────────────────────

def test_channel_dimension_names_follow_dim_pattern(sample_numpy_data):
    # Arrange
    result = to_sktime_df(sample_numpy_data)
    # Act
    # Assert
    for i in range(sample_numpy_data.shape[1]):
        assert result.iloc[0, i].name == f"dim_{i}"


# ── data preservation ──────────────────────────────────────────

def test_data_values_preserved_after_conversion(sample_numpy_data):
    # Arrange
    sample_idx, channel_idx = 0, 0
    original_data = sample_numpy_data[sample_idx, channel_idx, :]
    # Act
    result = to_sktime_df(sample_numpy_data)
    converted_data = result.iloc[sample_idx, channel_idx].values
    # Assert
    assert np.allclose(original_data, converted_data)


# ── shape tests ────────────────────────────────────────────────

def test_multiple_samples_conversion_returns_correct_sample_count():
    # Arrange
    X = np.random.rand(20, 4, 100)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert len(result) == 20


def test_multiple_samples_conversion_returns_correct_channel_count():
    # Arrange
    X = np.random.rand(20, 4, 100)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert len(result.columns) == 4


def test_single_sample_conversion_returns_one_row():
    # Arrange
    X = np.random.rand(1, 2, 30)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert len(result) == 1


def test_single_channel_conversion_returns_five_rows():
    # Arrange
    X = np.random.rand(5, 1, 25)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert len(result) == 5


# ── torch type tests ───────────────────────────────────────────

def test_float_tensor_returns_dataframe():
    # Arrange
    X_float = torch.rand(3, 2, 10)
    # Act
    result_float = to_sktime_df(X_float)
    # Assert
    assert isinstance(result_float, pd.DataFrame)


def test_double_tensor_converts_cell_to_float64():
    # Arrange
    X_double = torch.rand(3, 2, 10, dtype=torch.double)
    # Act
    result_double = to_sktime_df(X_double)
    channel = result_double.iloc[0, 0]
    # Assert
    assert channel.dtype == np.float64


def test_torch_and_numpy_inputs_produce_same_shape():
    # Arrange
    X_torch = torch.rand(4, 3, 20)
    X_numpy = X_torch.numpy()
    # Act
    result_torch = to_sktime_df(X_torch)
    result_numpy = to_sktime_df(X_numpy)
    # Assert
    assert result_torch.shape == result_numpy.shape


# ── invalid input tests ────────────────────────────────────────

def test_list_input_raises_value_error():
    # Arrange
    # Act
    ctx = pytest.raises(
        ValueError,
        match="Input X must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame",
    )
    # Assert
    with ctx:
        to_sktime_df([[1, 2, 3], [4, 5, 6]])


def test_string_input_raises_value_error():
    # Arrange
    # Act
    ctx = pytest.raises(
        ValueError,
        match="Input X must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame",
    )
    # Assert
    with ctx:
        to_sktime_df("invalid")


def test_dict_input_raises_value_error():
    # Arrange
    # Act
    ctx = pytest.raises(
        ValueError,
        match="Input X must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame",
    )
    # Assert
    with ctx:
        to_sktime_df({"data": [1, 2, 3]})


# ── edge cases ─────────────────────────────────────────────────

def test_two_dimensional_input_returns_dataframe():
    # Arrange
    X_2d = np.random.rand(10, 50)
    # Act
    result = to_sktime_df(X_2d)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_empty_array_returns_empty_dataframe():
    # Arrange
    X_empty = np.array([]).reshape(0, 2, 10)
    # Act
    result = to_sktime_df(X_empty)
    # Assert
    assert len(result) == 0


def test_zero_sequence_length_returns_correct_sample_count():
    # Arrange
    X_zero_seq = np.random.rand(3, 2, 0)
    # Act
    result = to_sktime_df(X_zero_seq)
    # Assert
    assert len(result) == 3


def test_large_dataset_returns_correct_sample_count():
    # Arrange
    X_large = np.random.rand(100, 10, 500)
    # Act
    result = to_sktime_df(X_large)
    # Assert
    assert len(result) == 100


# ── consistency tests ──────────────────────────────────────────

def test_data_structure_consistent_across_all_cells(sample_numpy_data):
    # Arrange
    n_samples, n_channels, seq_len = sample_numpy_data.shape
    # Act
    result = to_sktime_df(sample_numpy_data)
    # Assert
    assert result.shape == (n_samples, n_channels)


def test_example_from_docstring_returns_dataframe():
    # Arrange
    X_np = np.random.rand(64, 160, 1024)
    # Act
    sktime_df = to_sktime_df(X_np)
    # Assert
    assert isinstance(sktime_df, pd.DataFrame)


def test_single_timepoint_data_returns_correct_channel_count():
    # Arrange
    X = np.random.rand(5, 3, 1)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert len(result.columns) == 3


def test_numerical_precision_maintained_after_conversion():
    # Arrange
    X = np.array([[[1.123456789, 2.987654321, 3.141592653]]]).astype(np.float64)
    expected = np.array([1.123456789, 2.987654321, 3.141592653])
    # Act
    result = to_sktime_df(X)
    channel = result.iloc[0, 0]
    # Assert
    assert np.allclose(channel.values, expected, rtol=0, atol=1e-9)


def test_channel_indexing_columns_are_sequential_integers():
    # Arrange
    X = np.random.rand(5, 4, 20)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert list(result.columns) == [0, 1, 2, 3]


def test_original_data_not_modified_by_conversion():
    # Arrange
    X_original = np.random.rand(3, 2, 10)
    X_copy = X_original.copy()
    # Act
    to_sktime_df(X_original)
    # Assert
    assert np.array_equal(X_original, X_copy)


def test_large_data_returns_correct_shape():
    # Arrange
    X = np.random.rand(50, 20, 1000)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert result.shape == (50, 20)


def test_short_sequence_first_cell_has_length_5():
    # Arrange
    X_short = np.random.rand(3, 2, 5)
    # Act
    result_short = to_sktime_df(X_short)
    # Assert
    assert result_short.iloc[0, 0].shape[0] == 5


def test_long_sequence_first_cell_has_length_2000():
    # Arrange
    X_long = np.random.rand(3, 2, 2000)
    # Act
    result_long = to_sktime_df(X_long)
    # Assert
    assert result_long.iloc[0, 0].shape[0] == 2000


def test_torch_gradient_tensor_returns_dataframe():
    # Arrange
    X = torch.rand(3, 2, 10, requires_grad=True)
    # Act
    result = to_sktime_df(X)
    # Assert
    assert isinstance(result, pd.DataFrame)
