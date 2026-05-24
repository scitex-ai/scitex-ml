#!/usr/bin/env python3
# Time-stamp: "2025-06-01 13:05:00 (ywatanabe)"
# File: ./tests/scitex/ai/clustering/test__pca.py

"""Tests for scitex_ml.clustering._pca module."""

import pytest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from scitex_ml.clustering import pca


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    # Create two datasets with different distributions
    data1 = np.random.randn(100, 10) + np.array([1] * 10)
    data2 = np.random.randn(100, 10) + np.array([-1] * 10)

    # Create labels
    labels1 = ["A"] * 50 + ["B"] * 50
    labels2 = ["C"] * 50 + ["D"] * 50

    return {
        "single": (data1, labels1),
        "multiple": ([data1, data2], [labels1, labels2]),
    }


# ---------------------------------------------------------------------------
# Tests split from test_pca_single_dataset
# ---------------------------------------------------------------------------


def test_pca_single_dataset_returns_non_null_figure(sample_data):
    """Test PCA with a single dataset returns a non-null figure."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], title="Test PCA Single"
    )

    # Assert
    assert fig is not None

    plt.close(fig)


def test_pca_single_dataset_figure_is_matplotlib_type(sample_data):
    """Test PCA with a single dataset returns a matplotlib Figure."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], title="Test PCA Single"
    )

    # Assert
    assert isinstance(fig, plt.Figure)

    plt.close(fig)


def test_pca_single_dataset_model_is_sklearn_pca(sample_data):
    """Test PCA with a single dataset returns an sklearn PCA model."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], title="Test PCA Single"
    )

    # Assert
    assert isinstance(pca_model, SklearnPCA)

    plt.close(fig)


def test_pca_single_dataset_model_has_two_components(sample_data):
    """Test PCA with a single dataset uses 2 PCA components."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], title="Test PCA Single"
    )

    # Assert
    assert pca_model.n_components == 2

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_multiple_datasets
# ---------------------------------------------------------------------------


def test_pca_multiple_datasets_returns_non_null_figure(sample_data):
    """Test PCA with multiple datasets returns a non-null figure."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=data_list,
        labels_all=labels_list,
        title="Test PCA Multiple",
        axes_titles=["Dataset 1", "Dataset 2"],
    )

    # Assert
    assert fig is not None

    plt.close(fig)


def test_pca_multiple_datasets_figure_is_matplotlib_type(sample_data):
    """Test PCA with multiple datasets returns a matplotlib Figure."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=data_list,
        labels_all=labels_list,
        title="Test PCA Multiple",
        axes_titles=["Dataset 1", "Dataset 2"],
    )

    # Assert
    assert isinstance(fig, plt.Figure)

    plt.close(fig)


def test_pca_multiple_datasets_model_is_sklearn_pca(sample_data):
    """Test PCA with multiple datasets returns an sklearn PCA model."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=data_list,
        labels_all=labels_list,
        title="Test PCA Multiple",
        axes_titles=["Dataset 1", "Dataset 2"],
    )

    # Assert
    assert isinstance(pca_model, SklearnPCA)

    plt.close(fig)


def test_pca_multiple_datasets_has_sufficient_axes_count(sample_data):
    """Test PCA with multiple datasets creates enough axes."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=data_list,
        labels_all=labels_list,
        title="Test PCA Multiple",
        axes_titles=["Dataset 1", "Dataset 2"],
    )

    # Assert
    axes = fig.get_axes()
    assert len(axes) >= len(data_list)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_with_super_imposed
# ---------------------------------------------------------------------------


def test_pca_with_super_imposed_returns_non_null_figure(sample_data):
    """Test PCA with superimposed plot returns a non-null figure."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=data_list,
        labels_all=labels_list,
        add_super_imposed=True,
        title="Test PCA Superimposed",
    )

    # Assert
    assert fig is not None

    plt.close(fig)


def test_pca_with_super_imposed_has_extra_axis_for_overlay(sample_data):
    """Test PCA with superimposed plot adds one extra axis for overlay."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=data_list,
        labels_all=labels_list,
        add_super_imposed=True,
        title="Test PCA Superimposed",
    )

    # Assert
    axes = fig.get_axes()
    assert len(axes) == len(data_list) + 1

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_visual_parameters (single assert, just renamed)
# ---------------------------------------------------------------------------


def test_pca_visual_parameters_returns_non_null_figure(sample_data):
    """Test PCA with custom visual parameters returns a non-null figure."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], alpha=0.5, s=10, palette="coolwarm"
    )

    # Assert
    assert fig is not None

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_independent_legend (single assert, renamed, kept skip)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(True, reason="Source code has bug with use_independent_legend=True on single dataset (axes not iterable)")
def test_pca_independent_legend_returns_non_null_figure(sample_data):
    """Test PCA with independent legend returns a non-null figure."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], use_independent_legend=True
    )

    # Assert
    assert fig is not None

    # Cleanup
    if legend_figs is not None:
        for leg_fig in legend_figs:
            plt.close(leg_fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_label_encoding (single assert, renamed)
# ---------------------------------------------------------------------------


def test_pca_label_encoding_returns_non_null_figure(sample_data):
    """Test that labels are properly encoded and figure is returned."""
    # Arrange
    data, _ = sample_data["single"]
    labels = ["Group_" + str(i % 3) for i in range(len(data))]

    # Act
    fig, legend_figs, pca_model = pca(data_all=[data], labels_all=[labels])

    # Assert
    assert fig is not None

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_transform_consistency
# ---------------------------------------------------------------------------


def test_pca_transform_consistency_first_dataset_transformed_to_2d(sample_data):
    """Test that the first dataset is transformed to 2 dimensions."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(data_all=data_list, labels_all=labels_list)

    # Assert
    transformed1 = pca_model.transform(data_list[0])
    assert transformed1.shape == (len(data_list[0]), 2)

    plt.close(fig)


def test_pca_transform_consistency_second_dataset_transformed_to_2d(sample_data):
    """Test that the second dataset is also transformed to 2 dimensions."""
    # Arrange
    data_list, labels_list = sample_data["multiple"]

    # Act
    fig, legend_figs, pca_model = pca(data_all=data_list, labels_all=labels_list)

    # Assert
    transformed2 = pca_model.transform(data_list[1])
    assert transformed2.shape == (len(data_list[1]), 2)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_empty_data (single assert, renamed)
# ---------------------------------------------------------------------------


def test_pca_empty_data_raises_error_on_empty_input():
    """Test PCA with empty data raises a ValueError or IndexError."""
    # Arrange
    # Act
    ctx = pytest.raises((ValueError, IndexError))
    # Assert
    with ctx:
        pca(data_all=[], labels_all=[])


# ---------------------------------------------------------------------------
# Tests split from test_pca_mismatched_lengths (single assert, renamed)
# ---------------------------------------------------------------------------


def test_pca_mismatched_lengths_raises_error_on_length_mismatch(sample_data):
    """Test PCA with mismatched data and label lengths raises an error."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    ctx = pytest.raises((AssertionError, ValueError))
    # Assert
    with ctx:
        pca(
            data_all=[data],
            labels_all=[labels[:50]],  # Wrong length
        )


# ---------------------------------------------------------------------------
# Tests split from test_pca_numpy_array_input (single assert, renamed)
# ---------------------------------------------------------------------------


def test_pca_numpy_array_input_returns_non_null_figure(sample_data):
    """Test PCA with numpy array inputs wrapped in lists returns a figure."""
    # Arrange
    data, labels = sample_data["single"]
    data_array = np.array(data)
    labels_array = np.array(labels)

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data_array], labels_all=[labels_array]
    )

    # Assert
    assert fig is not None

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_different_dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples,n_features", [(50, 5), (100, 20), (200, 50)])
def test_pca_different_dimensions_returns_non_null_figure(n_samples, n_features):
    """Test PCA with different data dimensions returns a non-null figure."""
    # Arrange
    data = np.random.randn(n_samples, n_features)
    labels = ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)

    # Act
    fig, legend_figs, pca_model = pca(data_all=[data], labels_all=[labels])

    # Assert
    assert fig is not None

    plt.close(fig)


@pytest.mark.parametrize("n_samples,n_features", [(50, 5), (100, 20), (200, 50)])
def test_pca_different_dimensions_model_has_two_components(n_samples, n_features):
    """Test PCA with different data dimensions still yields 2-component model."""
    # Arrange
    data = np.random.randn(n_samples, n_features)
    labels = ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)

    # Act
    fig, legend_figs, pca_model = pca(data_all=[data], labels_all=[labels])

    # Assert
    assert pca_model.n_components == 2

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_axes_labels
# ---------------------------------------------------------------------------


def test_pca_axes_labels_has_non_null_suptitle(sample_data):
    """Test that PCA figure has a non-null suptitle."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], title="Test Title"
    )

    # Assert
    assert fig._suptitle is not None

    plt.close(fig)


def test_pca_axes_labels_suptitle_contains_given_title_text(sample_data):
    """Test that the PCA figure suptitle contains the specified title text."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], title="Test Title"
    )

    # Assert
    assert "Test Title" in fig._suptitle.get_text()

    plt.close(fig)


def test_pca_axes_labels_returns_non_null_figure(sample_data):
    """Test that successful completion returns a non-null figure."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(
        data_all=[data], labels_all=[labels], title="Test Title"
    )

    # Assert
    assert fig is not None

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_natural_sorting (single assert, renamed)
# ---------------------------------------------------------------------------


def test_pca_natural_sorting_returns_non_null_figure():
    """Test that naturally sorted labels still produce a valid figure."""
    # Arrange
    data = np.random.randn(100, 10)
    labels = ["Label_1", "Label_10", "Label_2", "Label_20"] * 25

    # Act
    fig, legend_figs, pca_model = pca(data_all=[data], labels_all=[labels])

    # Assert
    assert fig is not None

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests split from test_pca_explained_variance
# ---------------------------------------------------------------------------


def test_pca_explained_variance_model_has_variance_ratio_attribute(sample_data):
    """Test that the PCA model has explained_variance_ratio_ attribute."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(data_all=[data], labels_all=[labels])

    # Assert
    assert hasattr(pca_model, "explained_variance_ratio_")

    plt.close(fig)


def test_pca_explained_variance_ratio_has_two_entries(sample_data):
    """Test that explained_variance_ratio_ has 2 entries for 2 components."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(data_all=[data], labels_all=[labels])

    # Assert
    assert len(pca_model.explained_variance_ratio_) == 2

    plt.close(fig)


def test_pca_explained_variance_ratio_sum_is_at_most_one(sample_data):
    """Test that the sum of explained_variance_ratio_ does not exceed 1.0."""
    # Arrange
    data, labels = sample_data["single"]

    # Act
    fig, legend_figs, pca_model = pca(data_all=[data], labels_all=[labels])

    # Assert
    assert np.sum(pca_model.explained_variance_ratio_) <= 1.0

    plt.close(fig)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])


# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/clustering/_pca.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-05-14 00:58:26 (ywatanabe)"
#
# import matplotlib.pyplot as plt
# import scitex
# import numpy as np
# import seaborn as sns
# from natsort import natsorted
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import LabelEncoder
#
#
# def pca(
#     data_all,
#     labels_all,
#     axes_titles=None,
#     title="PCA Clustering",
#     alpha=0.1,
#     s=3,
#     use_independent_legend=False,
#     add_super_imposed=False,
#     palette="viridis",
# ):
#     assert len(data_all) == len(labels_all)
#
#     if isinstance(data_all, list):
#         data_all = list(data_all)
#         labels_all = list(labels_all)
#
#     le = LabelEncoder()
#     # le.fit(np.hstack(labels_all))
#     le.fit(natsorted(np.hstack(labels_all)))
#     labels_all = [le.transform(labels) for labels in labels_all]
#
#     pca_model = PCA(n_components=2)
#
#     ncols = len(data_all) + 1 if add_super_imposed else len(data_all)
#     share = True if ncols > 1 else False
#     fig, axes = plt.subplots(ncols=ncols, sharex=share, sharey=share)
#
#     fig.suptitle(title)
#     fig.supxlabel("PCA 1")
#     fig.supylabel("PCA 2")
#
#     for ii, (data, labels) in enumerate(zip(data_all, labels_all)):
#         if ii == 0:
#             _pca = pca_model.fit(data)
#             embedding = _pca.transform(data)
#         else:
#             embedding = pca_model.transform(data)
#
#         if ncols == 1:
#             ax = axes
#         else:
#             ax = axes[ii + 1] if add_super_imposed else axes[ii]
#
#         sns.scatterplot(
#             x=embedding[:, 0],
#             y=embedding[:, 1],
#             hue=le.inverse_transform(labels),
#             ax=ax,
#             palette=palette,
#             s=s,
#             alpha=alpha,
#         )
#
#         ax.set_box_aspect(1)
#
#         if axes_titles is not None:
#             ax.set_title(axes_titles[ii])
#
#         if not use_independent_legend:
#             ax.legend(loc="upper left")
#
#         if add_super_imposed:
#             axes[0].set_title("Superimposed")
#             axes[0].set_aspect("equal")
#
#             sns.scatterplot(
#                 x=embedding[:, 0],
#                 y=embedding[:, 1],
#                 hue=le.inverse_transform(labels),
#                 ax=axes[0],
#                 palette=palette,
#                 legend="full" if ii == 0 else False,
#                 s=s,
#                 alpha=alpha,
#             )
#
#     if not use_independent_legend:
#         return fig, None, pca_model
#
#     elif use_independent_legend:
#         legend_figs = []
#         for i, ax in enumerate(axes):
#             legend = ax.get_legend()
#             if legend:
#                 legend_fig = plt.figure(figsize=(3, 2))
#                 new_legend = legend_fig.gca().legend(
#                     handles=legend.legendHandles,
#                     labels=legend.texts,
#                     loc="center",
#                 )
#                 legend_fig.canvas.draw()
#                 legend_filename = f"legend_{i}.png"
#                 legend_fig.savefig(legend_filename, bbox_inches="tight")
#                 legend_figs.append(legend_fig)
#                 plt.close(legend_fig)
#
#         for ax in axes:
#             ax.legend_ = None
#             # ax.remove_legend()
#             return fig, legend_figs, pca_model

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/clustering/_pca.py
# --------------------------------------------------------------------------------
