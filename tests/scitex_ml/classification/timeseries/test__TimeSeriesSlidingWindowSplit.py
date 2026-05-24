"""Smoke test for scitex_ml.classification.timeseries._TimeSeriesSlidingWindowSplit.

The test imports the target module and asserts the returned module
matches its dotted name. Renames, broken peer deps, or missing
optional deps all surface here as red, not as a silent skip.

If a module legitimately requires an optional dep, that dep should
be lazy-imported inside the function bodies — not at module top.
"""

import importlib


def test_module_imports_under_expected_dotted_name():
    """Imported module's `__name__` matches the dotted target."""
    # Arrange
    target = "scitex_ml.classification.timeseries._TimeSeriesSlidingWindowSplit"
    # Act
    mod = importlib.import_module(target)
    # Assert
    assert mod.__name__ == target
