"""Tests for scitex_ml.classification.reporters._mixins._constants module."""

import pytest

pytest.importorskip("scitex_ml")


def test_filename_patterns_dict_contains_expected_keys():
    # Arrange
    from scitex_ml.classification.reporters._mixins._constants import FILENAME_PATTERNS

    expected_keys = [
        "fold_metric_with_value",
        "fold_metric",
        "confusion_matrix_csv",
        "confusion_matrix_jpg",
        "classification_report",
        "roc_curve_csv",
        "pr_curve_csv",
        "y_true",
        "y_pred",
        "y_proba",
        "metrics_summary",
        "feature_importance_json",
    ]
    # Act
    missing = [k for k in expected_keys if k not in FILENAME_PATTERNS]
    # Assert
    assert not missing, f"FILENAME_PATTERNS missing keys: {missing}"


def test_fold_dir_prefix_matches_format_pattern():
    # Arrange
    from scitex_ml.classification.reporters._mixins._constants import FOLD_DIR_PREFIX_PATTERN
    # Act
    formatted = FOLD_DIR_PREFIX_PATTERN.format(fold=7)
    # Assert
    assert formatted == "fold_07"


def test_fold_file_prefix_matches_format_pattern():
    # Arrange
    from scitex_ml.classification.reporters._mixins._constants import FOLD_FILE_PREFIX_PATTERN
    # Act
    formatted = FOLD_FILE_PREFIX_PATTERN.format(fold=7)
    # Assert
    assert formatted == "fold-07"
