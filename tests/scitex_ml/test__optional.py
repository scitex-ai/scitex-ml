"""Tests for scitex_ml._optional.missing_optional_dependency."""

from scitex_ml._optional import INSTALL_HINT, missing_optional_dependency


def _original_error():
    return ImportError("No module named 'torch'", name="torch")


def test_missing_optional_dependency_returns_import_error():
    """The helper builds an ImportError the caller can raise."""
    # Arrange
    exc = _original_error()
    # Act
    result = missing_optional_dependency("scitex_ml.loss", exc)
    # Assert
    assert isinstance(result, ImportError)


def test_missing_optional_dependency_message_names_install_line():
    """The message carries the single `[all]` install line."""
    # Arrange
    exc = _original_error()
    # Act
    message = str(missing_optional_dependency("scitex_ml.loss", exc))
    # Assert
    assert INSTALL_HINT in message


def test_install_hint_is_the_all_extra():
    """The only install line users are told is scitex-ml[all]."""
    # Arrange
    expected = 'pip install "scitex-ml[all]"'
    # Act
    actual = INSTALL_HINT
    # Assert
    assert actual == expected


def test_missing_optional_dependency_message_names_feature():
    """The message names the scitex-ml capability that needs the dependency."""
    # Arrange
    exc = _original_error()
    # Act
    message = str(missing_optional_dependency("scitex_ml.loss", exc))
    # Assert
    assert "scitex_ml.loss" in message


def test_missing_optional_dependency_message_names_missing_module():
    """The message names the module whose import failed."""
    # Arrange
    exc = _original_error()
    # Act
    message = str(missing_optional_dependency("scitex_ml.loss", exc))
    # Assert
    assert "'torch'" in message


def test_missing_optional_dependency_keeps_module_name_attribute():
    """ImportError.name is preserved so callers can introspect it."""
    # Arrange
    exc = _original_error()
    # Act
    result = missing_optional_dependency("scitex_ml.loss", exc)
    # Assert
    assert result.name == "torch"


def test_missing_optional_dependency_message_names_distribution_when_given():
    """A distinct PyPI distribution name is included when supplied."""
    # Arrange
    exc = ImportError(
        "No module named 'scitex_seizure_metrics'", name="scitex_seizure_metrics"
    )
    # Act
    message = str(
        missing_optional_dependency(
            "scitex_ml.metrics.seizure", exc, distribution="scitex-seizure-metrics"
        )
    )
    # Assert
    assert "scitex-seizure-metrics" in message
