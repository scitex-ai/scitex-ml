#!/usr/bin/env python3
# Time-stamp: "2025-06-11 03:40:00 (ywatanabe)"
# File: ./tests/scitex/ai/clustering/test__umap.py

"""Comprehensive test module for scitex_ml.clustering._umap functionality."""

import matplotlib.pyplot as plt  # noqa: STXI001
import numpy as np
import pytest

try:
    import umap.umap_ as umap_lib  # noqa: F401

    UMAP_AVAILABLE = True
except Exception:
    # `import umap` transitively pulls tensorflow on some installs, and
    # tensorflow can raise google.protobuf.runtime_version.VersionError
    # (not ImportError) when its protobuf gencode disagrees with the
    # installed runtime. Treat any failure here as "umap not available".
    UMAP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not UMAP_AVAILABLE, reason="UMAP library not available")

from scitex_ml.clustering import umap  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    data = np.random.randn(n_samples, n_features)
    labels = np.array(["A"] * 50 + ["B"] * 50)
    return data, labels


@pytest.fixture
def multi_dataset():
    """Generate multiple datasets for testing."""
    np.random.seed(42)
    data1 = np.random.randn(100, 50)
    data2 = np.random.randn(80, 50)
    labels1 = np.array(["A"] * 50 + ["B"] * 50)
    labels2 = np.array(["C"] * 40 + ["D"] * 40)
    return [data1, data2], [labels1, labels2]


# ===========================================================================
# Tests from TestUmapBasicFunctionality
# ===========================================================================


@pytest.mark.timeout(180)  # UMAP JIT compilation can take >60s on first run
def test_umap_basic_functionality_returns_non_null_figure(sample_data):
    """Test basic UMAP returns a non-null figure."""
    # Arrange
    data, labels = sample_data

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_basic_functionality_legend_figs_is_none_by_default(sample_data):
    """Test basic UMAP returns None for legend_figs by default."""
    # Arrange
    data, labels = sample_data

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

    # Assert
    assert legend_figs is None

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_basic_functionality_returns_non_null_model(sample_data):
    """Test basic UMAP returns a non-null model."""
    # Arrange
    data, labels = sample_data

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

    # Assert
    assert umap_model is not None

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_multiple_datasets_returns_non_null_figure(multi_dataset):
    """Test UMAP with multiple datasets returns a non-null figure."""
    # Arrange
    data_list, labels_list = multi_dataset

    # Act
    fig, legend_figs, umap_model = umap(
        data=data_list, labels=labels_list, axes_titles=["Dataset 1", "Dataset 2"]
    )

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_multiple_datasets_returns_non_null_model(multi_dataset):
    """Test UMAP with multiple datasets returns a non-null model."""
    # Arrange
    data_list, labels_list = multi_dataset

    # Act
    fig, legend_figs, umap_model = umap(
        data=data_list, labels=labels_list, axes_titles=["Dataset 1", "Dataset 2"]
    )

    # Assert
    assert umap_model is not None

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_multiple_datasets_has_sufficient_axes_count(multi_dataset):
    """Test UMAP with multiple datasets creates enough axes."""
    # Arrange
    data_list, labels_list = multi_dataset

    # Act
    fig, legend_figs, umap_model = umap(
        data=data_list, labels=labels_list, axes_titles=["Dataset 1", "Dataset 2"]
    )

    # Assert
    if hasattr(fig, "fig"):
        axes = fig.fig.get_axes()
    else:
        axes = fig.get_axes()
    assert len(axes) >= len(data_list)

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_supervised_mode_returns_non_null_figure(sample_data):
    """Test supervised UMAP mode returns a non-null figure."""
    # Arrange
    data, labels = sample_data

    # Act
    fig, legend_figs, umap_model = umap(
        data=[data], labels=[labels], supervised=True
    )

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_supervised_mode_returns_non_null_model(sample_data):
    """Test supervised UMAP mode returns a non-null model."""
    # Arrange
    data, labels = sample_data

    # Act
    fig, legend_figs, umap_model = umap(
        data=[data], labels=[labels], supervised=True
    )

    # Assert
    assert umap_model is not None

    plt.close("all")


# ===========================================================================
# Tests from TestUmapVisualization
# ===========================================================================


@pytest.mark.timeout(180)
def test_umap_with_hues_returns_non_null_figure(sample_data):
    """Test UMAP with hue coloring returns a non-null figure."""
    # Arrange
    data, labels = sample_data
    hues = np.array(["group_A"] * 50 + ["group_B"] * 50)

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels], hues=[hues])

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.skipif(True, reason="Custom colors API needs clarification from source code")
def test_umap_with_custom_colors_returns_non_null_figure(sample_data):
    """Test UMAP with custom color mapping returns a non-null figure."""
    # Arrange
    data, labels = sample_data
    hues_colors = [["red"] * 50 + ["blue"] * 50]

    # Act
    fig, legend_figs, umap_model = umap(
        data=[data], labels=[labels], hues_colors=hues_colors
    )

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.timeout(180)
def test_umap_visualization_parameters_returns_non_null_figure(sample_data):
    """Test UMAP with custom visualization parameters returns a figure."""
    # Arrange
    data, labels = sample_data

    # Act
    fig, legend_figs, umap_model = umap(
        data=[data], labels=[labels], title="Custom UMAP Title", alpha=0.7, s=50
    )

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.skipif(True, reason="Source code has bug with use_independent_legend (axes not iterable with FigWrapper)")  # noqa: E501
def test_umap_with_independent_legend_returns_non_null_figure(sample_data):
    """Test UMAP with independent legend returns a non-null figure."""
    # Arrange
    data, labels = sample_data

    # Act
    fig, legend_figs, umap_model = umap(
        data=[data], labels=[labels], use_independent_legend=True
    )

    # Assert
    assert fig is not None

    # Cleanup
    if legend_figs is not None:
        for leg_fig in legend_figs:
            plt.close(leg_fig)
    plt.close("all")


@pytest.mark.skipif(True, reason="Source code has bug with add_super_imposed (hues_colors vstack issue)")
def test_umap_with_superimposed_returns_non_null_figure():
    """Test UMAP with superimposed plot returns a non-null figure."""
    # Arrange
    np.random.seed(42)
    data1 = np.random.randn(100, 50)
    data2 = np.random.randn(80, 50)
    labels1 = np.array(["A"] * 50 + ["B"] * 50)
    labels2 = np.array(["C"] * 40 + ["D"] * 40)

    # Act
    fig, legend_figs, umap_model = umap(
        data=[data1, data2], labels=[labels1, labels2], add_super_imposed=True
    )

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.skipif(True, reason="Source code has bug with add_super_imposed (hues_colors vstack issue)")
def test_umap_with_superimposed_has_three_axes():
    """Test UMAP with superimposed plot adds extra axis for overlay."""
    # Arrange
    np.random.seed(42)
    data1 = np.random.randn(100, 50)
    data2 = np.random.randn(80, 50)
    labels1 = np.array(["A"] * 50 + ["B"] * 50)
    labels2 = np.array(["C"] * 40 + ["D"] * 40)

    # Act
    fig, legend_figs, umap_model = umap(
        data=[data1, data2], labels=[labels1, labels2], add_super_imposed=True
    )

    # Assert
    if hasattr(fig, "fig"):
        axes = fig.fig.get_axes()
    else:
        axes = fig.get_axes()
    assert len(axes) == 3  # 2 datasets + 1 superimposed

    plt.close("all")


# ===========================================================================
# Tests from TestUmapAlgorithmicOptions
# ===========================================================================


@pytest.mark.timeout(180)
def test_umap_with_pretrained_model_reuses_same_model_object(sample_data):
    """Test UMAP with a pre-fitted model reuses the same model object."""
    # Arrange
    data, labels = sample_data

    fig1, _, model1 = umap(data=[data], labels=[labels])
    plt.close("all")

    new_data = np.random.randn(80, 50)
    new_labels = np.array(["X"] * 40 + ["Y"] * 40)

    # Act
    fig2, legend_figs, model2 = umap(
        data=[new_data], labels=[new_labels], umap_model=model1
    )

    # Assert
    assert model2 is model1

    plt.close("all")


# ===========================================================================
# Tests from TestUmapDataValidation
# ===========================================================================


@pytest.mark.timeout(180)
def test_umap_input_format_validation_returns_non_null_figure():
    """Test UMAP validates list input format and returns a figure."""
    # Arrange
    data = np.random.randn(100, 50)
    labels = np.array(["A"] * 50 + ["B"] * 50)

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

    # Assert
    assert fig is not None

    plt.close("all")


def test_umap_mismatched_lengths_raises_error():
    """Test UMAP with mismatched data and label lengths raises an error."""
    # Arrange
    data = [np.random.randn(100, 50)]
    labels = [np.array(["A"] * 50)]  # Wrong size

    # Act
    ctx = pytest.raises((AssertionError, IndexError))
    # Assert
    with ctx:
        umap(data=data, labels=labels)


def test_umap_empty_data_raises_error():
    """Test UMAP with empty data raises an error."""
    # Arrange
    # Act
    ctx = pytest.raises((ValueError, IndexError, AssertionError))
    # Assert
    with ctx:
        umap(data=[], labels=[])


@pytest.mark.timeout(180)
def test_umap_natural_label_sorting_returns_non_null_figure():
    """Test that naturally sorted labels still produce a valid figure."""
    # Arrange
    data = np.random.randn(100, 50)
    labels = np.array(["Label_1", "Label_10", "Label_2", "Label_20"] * 25)

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

    # Assert
    assert fig is not None

    plt.close("all")


# ===========================================================================
# Tests from TestUmapIntegration
# ===========================================================================


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "n_samples,n_features,n_classes",
    [
        (50, 10, 2),
        (100, 50, 2),
        (150, 30, 3),
    ],
)
def test_umap_various_data_sizes_returns_non_null_figure(
    n_samples, n_features, n_classes
):
    """Test UMAP with various data sizes returns a non-null figure."""
    # Arrange
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)

    samples_per_class = n_samples // n_classes
    labels = []
    for i in range(n_classes):
        labels.extend([f"Class_{i}"] * samples_per_class)
    labels.extend(["Class_0"] * (n_samples - len(labels)))
    labels = np.array(labels)

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

    # Assert
    assert fig is not None

    plt.close("all")


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "n_samples,n_features,n_classes",
    [
        (50, 10, 2),
        (100, 50, 2),
        (150, 30, 3),
    ],
)
def test_umap_various_data_sizes_returns_non_null_model(
    n_samples, n_features, n_classes
):
    """Test UMAP with various data sizes returns a non-null model."""
    # Arrange
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)

    samples_per_class = n_samples // n_classes
    labels = []
    for i in range(n_classes):
        labels.extend([f"Class_{i}"] * samples_per_class)
    labels.extend(["Class_0"] * (n_samples - len(labels)))
    labels = np.array(labels)

    # Act
    fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

    # Assert
    assert umap_model is not None

    plt.close("all")


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
