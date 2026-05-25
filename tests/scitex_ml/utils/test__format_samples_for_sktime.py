import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scitex_ml.utils import format_samples_for_sktime
from scitex_ml.utils._format_samples_for_sktime import _format_a_sample_for_sktime


# ---------------------------------------------------------------------------
# _format_a_sample_for_sktime
# ---------------------------------------------------------------------------


def test_format_single_sample_returns_pandas_series():
    # Arrange
    x = np.random.rand(3, 100)
    # Act
    result = _format_a_sample_for_sktime(x)
    # Assert
    assert isinstance(result, pd.Series)


def test_format_single_sample_has_correct_series_length():
    # Arrange
    x = np.random.rand(3, 100)
    # Act
    result = _format_a_sample_for_sktime(x)
    # Assert
    assert len(result) == 3


def test_format_single_sample_preserves_dim_0_values():
    # Arrange
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # Act
    result = _format_a_sample_for_sktime(x)
    # Assert
    assert list(result["dim_0"].values) == [1, 2, 3]


def test_format_single_sample_preserves_dim_1_values():
    # Arrange
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # Act
    result = _format_a_sample_for_sktime(x)
    # Assert
    assert list(result["dim_1"].values) == [4, 5, 6]


def test_series_dim_0_name_matches_expected_pattern():
    # Arrange
    x = np.random.rand(4, 30)
    # Act
    result = _format_a_sample_for_sktime(x)
    # Assert
    assert result["dim_0"].name == "dim_0"


def test_series_dim_1_name_matches_expected_pattern():
    # Arrange
    x = np.random.rand(4, 30)
    # Act
    result = _format_a_sample_for_sktime(x)
    # Assert
    assert result["dim_1"].name == "dim_1"


# ---------------------------------------------------------------------------
# format_samples_for_sktime — numpy input
# ---------------------------------------------------------------------------


def test_format_numpy_samples_returns_pandas_dataframe():
    # Arrange
    X = np.random.rand(10, 5, 50)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_format_numpy_samples_has_correct_row_count():
    # Arrange
    X = np.random.rand(10, 5, 50)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert len(result) == 10


def test_format_numpy_samples_has_correct_column_count():
    # Arrange
    X = np.random.rand(10, 5, 50)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert result.shape[1] == 5


# ---------------------------------------------------------------------------
# format_samples_for_sktime — torch input
# ---------------------------------------------------------------------------


def test_format_torch_samples_returns_pandas_dataframe():
    # Arrange
    X = torch.randn(20, 3, 100)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_format_torch_samples_has_correct_row_count():
    # Arrange
    X = torch.randn(20, 3, 100)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert len(result) == 20


def test_format_torch_samples_has_correct_column_count():
    # Arrange
    X = torch.randn(20, 3, 100)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert result.shape[1] == 3


# ---------------------------------------------------------------------------
# torch ↔ numpy equivalence
# ---------------------------------------------------------------------------


def test_torch_numpy_inputs_produce_identical_dataframes():
    # Arrange
    X_torch = torch.randn(5, 2, 30, dtype=torch.float64)
    X_numpy = X_torch.numpy()
    # Act
    result_torch = format_samples_for_sktime(X_torch)
    result_numpy = format_samples_for_sktime(X_numpy)
    # Assert
    assert result_torch.shape == result_numpy.shape and list(result_torch.columns) == list(result_numpy.columns) and all(
        np.allclose(result_torch.loc[i, col].values, result_numpy.loc[i, col].values, equal_nan=True)
        for i in range(len(result_torch))
        for col in result_torch.columns
    )


def test_torch_tensor_converts_cell_dtype_to_float64():
    # Arrange
    X = torch.randn(5, 3, 20).float()
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert result.loc[0, result.columns[0]].dtype == np.float64


# ---------------------------------------------------------------------------
# single channel
# ---------------------------------------------------------------------------


def test_single_channel_data_has_correct_shape():
    # Arrange
    X = np.random.rand(15, 1, 200)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert result.shape == (15, 1)


def test_single_channel_data_has_dim_0_column():
    # Arrange
    X = np.random.rand(15, 1, 200)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert "dim_0" in result.columns


# ---------------------------------------------------------------------------
# many channels
# ---------------------------------------------------------------------------


def test_many_channels_has_correct_column_count():
    # Arrange
    n_channels = 100
    X = np.random.rand(5, n_channels, 50)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert result.shape[1] == n_channels


def test_column_names_match_expected_dim_i_pattern():
    # Arrange
    X = np.random.rand(10, 7, 30)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert list(result.columns) == [f"dim_{i}" for i in range(7)]


# ---------------------------------------------------------------------------
# empty data
# ---------------------------------------------------------------------------


def test_empty_data_returns_empty_pandas_dataframe():
    # Arrange
    X = np.array([]).reshape(0, 5, 100)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_empty_data_has_zero_rows():
    # Arrange
    X = np.array([]).reshape(0, 5, 100)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert len(result) == 0


# ---------------------------------------------------------------------------
# single-sample 3D array
# ---------------------------------------------------------------------------


def test_single_sample_3d_array_returns_pandas_dataframe():
    # Arrange
    X = np.random.rand(1, 3, 50)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_single_sample_3d_array_has_correct_row_count():
    # Arrange
    X = np.random.rand(1, 3, 50)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert len(result) == 1


def test_single_sample_3d_array_has_correct_shape():
    # Arrange
    X = np.random.rand(1, 3, 50)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert result.shape == (1, 3)


# ---------------------------------------------------------------------------
# data integrity — specific values preserved
# ---------------------------------------------------------------------------


def test_first_sample_dim_0_values_are_preserved():
    # Arrange
    X = np.arange(24).reshape(2, 3, 4).astype(np.float64)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert list(result.loc[0, "dim_0"].values) == [0, 1, 2, 3]


def test_first_sample_dim_1_values_are_preserved():
    # Arrange
    X = np.arange(24).reshape(2, 3, 4).astype(np.float64)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert list(result.loc[0, "dim_1"].values) == [4, 5, 6, 7]


def test_first_sample_dim_2_values_are_preserved():
    # Arrange
    X = np.arange(24).reshape(2, 3, 4).astype(np.float64)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert list(result.loc[0, "dim_2"].values) == [8, 9, 10, 11]


def test_second_sample_dim_0_values_are_preserved():
    # Arrange
    X = np.arange(24).reshape(2, 3, 4).astype(np.float64)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert list(result.loc[1, "dim_0"].values) == [12, 13, 14, 15]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_single_nan_value_preserved_through_conversion():
    # Arrange
    X = np.random.rand(5, 2, 25)
    X[0, 0, 5] = np.nan
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert np.isnan(result.loc[0, "dim_0"].iloc[5])


def test_multiple_nan_values_preserved_through_conversion():
    # Arrange
    X = np.random.rand(5, 2, 25)
    X[2, 1, 10:15] = np.nan
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert np.all(np.isnan(result.loc[2, "dim_1"].iloc[10:15]))


# ---------------------------------------------------------------------------
# Inf handling
# ---------------------------------------------------------------------------


def test_positive_inf_value_preserved_through_conversion():
    # Arrange
    X = np.random.rand(3, 2, 20)
    X[1, 0, 0] = np.inf
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert np.isinf(result.loc[1, "dim_0"].iloc[0])


def test_negative_inf_value_preserved_through_conversion():
    # Arrange
    X = np.random.rand(3, 2, 20)
    X[2, 1, -1] = -np.inf
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert np.isinf(result.loc[2, "dim_1"].iloc[-1])


# ---------------------------------------------------------------------------
# DataFrame index
# ---------------------------------------------------------------------------


def test_dataframe_index_is_sequential_range():
    # Arrange
    n_samples = 25
    X = np.random.rand(n_samples, 5, 40)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert list(result.index) == list(range(n_samples))


# ---------------------------------------------------------------------------
# CUDA tensor handling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_returns_pandas_dataframe():
    # Arrange
    X = torch.randn(10, 3, 50).cuda()
    # Act
    result = format_samples_for_sktime(X.cpu())
    # Assert
    assert isinstance(result, pd.DataFrame)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_has_correct_shape():
    # Arrange
    X = torch.randn(10, 3, 50).cuda()
    # Act
    result = format_samples_for_sktime(X.cpu())
    # Assert
    assert result.shape == (10, 3)


# ---------------------------------------------------------------------------
# large array
# ---------------------------------------------------------------------------


def test_large_array_has_correct_shape():
    # Arrange
    X = np.random.rand(100, 10, 1000)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert result.shape == (100, 10)


def test_large_array_cell_has_correct_length():
    # Arrange
    X = np.random.rand(100, 10, 1000)
    # Act
    result = format_samples_for_sktime(X)
    # Assert
    assert len(result.loc[50, "dim_5"]) == 1000
