#!/usr/bin/env python3
"""Per-edge integration + degradation tests (scitex-seizure-metrics edge).

This file mirrors the canonical figrecipe edge template in scitex-io. It
exercises the OPTIONAL collaborator edge between scitex-ml and the standalone
``scitex-seizure-metrics`` package.

The edge under test
-------------------
``from scitex_ml.metrics import seizure`` resolves to a thin re-export shim
(``scitex_ml/metrics/seizure/__init__.py``). scitex-seizure-metrics is an
OPTIONAL dependency of scitex-ml, declared behind the ``[seizure]`` extra:

    pip install scitex-ml[seizure]

When the standalone is installed, the shim aliases the full evaluation surface
(``detection``, ``forecasting``, ``AlarmPolicy``, ``MetricsReport``, ...) so
``scitex_ml.metrics.seizure.detection`` is the *same module object* as
``scitex_seizure_metrics.detection``. When it is ABSENT, importing the shim
must raise a clear, documented ``ImportError`` that points the user at the
install instruction — not an opaque traceback — while the non-seizure metrics
(``calc_bacc``, the pure-numpy ``calc_seizure_window_prediction_metrics``,
...) keep working untouched.

The two test kinds every optional edge should have
--------------------------------------------------
1. INTEGRATION (collaborator PRESENT): import the real shim and assert on the
   concrete aliasing it performs. Guarded with
   ``pytest.importorskip("scitex_seizure_metrics")`` so the suite stays green
   on minimal installs instead of erroring.

2. DEGRADATION (collaborator ABSENT): simulate the dependency being missing in
   a hermetic, reversible way (a fixture that snapshots ``sys.modules``,
   shadows ``scitex_seizure_metrics`` with ``None`` so a fresh import raises
   ImportError, evicts the cached shim, and restores everything on teardown),
   then assert the shim fails through its documented, caller-safe contract
   (a clear ``ImportError`` naming the ``[seizure]`` extra) and that the
   non-seizure metrics are unaffected.

Conventions honoured (so this stays a clean template):
  - One assertion per test (TQ007): shared, expensive setup is lifted into a
    fixture; each behaviour gets its own named, single-assert test.
  - Explicit Arrange / Act / Assert markers in every test (TQ002).
  - No ``monkeypatch`` / ``mocker`` (banned by repo convention): the
    seizure-absent fixture hand-swaps ``sys.modules`` and restores it on
    teardown.

Empirically verified contract
-----------------------------
present:  ``scitex_ml.metrics.seizure`` exposes ``__version__`` and aliases
          ``detection`` / ``forecasting`` / ``AlarmPolicy`` to the *same*
          objects as ``scitex_seizure_metrics``; the dotted submodule
          ``scitex_ml.metrics.seizure.detection`` is registered in
          ``sys.modules`` and is identical to ``scitex_seizure_metrics.detection``.
absent:   ``import scitex_ml.metrics.seizure`` raises ``ImportError`` whose
          message contains "scitex-seizure-metrics" and "pip install
          scitex-ml[seizure]"; ``scitex_ml.metrics.calc_bacc`` and the
          pure-numpy ``calc_seizure_window_prediction_metrics`` still import
          and run.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ===========================================================================
# 1. INTEGRATION  —  scitex-seizure-metrics PRESENT
# ===========================================================================
scitex_seizure_metrics = pytest.importorskip("scitex_seizure_metrics")


@pytest.fixture
def seizure_shim():
    """Import the scitex_ml.metrics.seizure shim once; yield (shim, impl)."""
    import scitex_seizure_metrics as impl

    seizure = importlib.import_module("scitex_ml.metrics.seizure")
    return seizure, impl


def test_seizure_shim_exposes_version(seizure_shim):
    """The shim re-exports the standalone's version string."""
    # Arrange
    seizure, impl = seizure_shim
    # Act
    version = seizure.__version__
    # Assert
    assert version == impl.__version__


def test_seizure_shim_aliases_detection_module(seizure_shim):
    """seizure.detection is the *same object* as the standalone's detection."""
    # Arrange
    seizure, impl = seizure_shim
    # Act
    same = seizure.detection is impl.detection
    # Assert
    assert same


def test_seizure_shim_aliases_forecasting_module(seizure_shim):
    """seizure.forecasting is the *same object* as the standalone's forecasting."""
    # Arrange
    seizure, impl = seizure_shim
    # Act
    same = seizure.forecasting is impl.forecasting
    # Assert
    assert same


def test_seizure_shim_reexports_alarm_policy(seizure_shim):
    """The AlarmPolicy class is re-exported and identical to the standalone's."""
    # Arrange
    seizure, impl = seizure_shim
    # Act
    same = seizure.AlarmPolicy is impl.AlarmPolicy
    # Assert
    assert same


def test_seizure_shim_registers_dotted_submodule(seizure_shim):
    """`scitex_ml.metrics.seizure.detection` is registered in sys.modules."""
    # Arrange
    _ = seizure_shim
    # Act
    registered = "scitex_ml.metrics.seizure.detection" in sys.modules
    # Assert
    assert registered


def test_seizure_shim_dotted_submodule_matches_standalone(seizure_shim):
    """The dotted submodule resolves to the standalone's module object."""
    # Arrange
    _, impl = seizure_shim
    # Act
    dotted = sys.modules["scitex_ml.metrics.seizure.detection"]
    # Assert
    assert dotted is impl.detection


def test_seizure_shim_all_lists_public_surface(seizure_shim):
    """The shim's __all__ advertises the documented evaluation surface."""
    # Arrange
    seizure, _ = seizure_shim
    expected = {
        "__version__",
        "detection",
        "forecasting",
        "AlarmPolicy",
        "MetricsReport",
    }
    # Act
    advertised = set(seizure.__all__)
    # Assert
    assert expected <= advertised


# ===========================================================================
# 2. DEGRADATION  —  scitex-seizure-metrics ABSENT
# ===========================================================================
@pytest.fixture
def seizure_metrics_absent():
    """Make ``import scitex_seizure_metrics`` fail for the test's duration.

    Hermetic and reversible:
      1. snapshot the whole ``sys.modules`` so teardown restores it exactly;
      2. evict ``scitex_seizure_metrics`` (its submodules) and the
         ``scitex_ml.metrics`` package + the seizure shim, so they re-run
         their import guards on the next import;
      3. shadow ``scitex_seizure_metrics`` with ``None`` so a *fresh*
         ``import scitex_seizure_metrics`` raises ImportError.

    Yields nothing; the test imports under the missing dependency itself.
    """
    import scitex_ml.metrics  # noqa: F401  (ensure importable before teardown)

    # 1. Full snapshot for an exact restore.
    snapshot = dict(sys.modules)

    # 2. Evict the standalone, the metrics package, and the seizure shim.
    def _to_evict(name: str) -> bool:
        return (
            name == "scitex_seizure_metrics"
            or name.startswith("scitex_seizure_metrics.")
            or name == "scitex_ml.metrics"
            or name.startswith("scitex_ml.metrics.")
        )

    for name in [n for n in list(sys.modules) if _to_evict(n)]:
        del sys.modules[name]

    # 3. Block a fresh import of the standalone.
    sys.modules["scitex_seizure_metrics"] = None  # type: ignore[assignment]

    try:
        yield
    finally:
        # Restore the exact pre-test module table.
        for name in list(sys.modules):
            if name not in snapshot:
                del sys.modules[name]
        sys.modules.update(snapshot)


def test_absent_fixture_blocks_the_import(seizure_metrics_absent):
    """Sanity: under the fixture, ``import scitex_seizure_metrics`` fails."""
    # Arrange
    _ = seizure_metrics_absent
    # Act
    module_name = "scitex_seizure_metrics"
    # Assert
    with pytest.raises(ImportError):
        importlib.import_module(module_name)


def test_seizure_shim_raises_importerror_when_absent(seizure_metrics_absent):
    """Importing the shim degrades to a clear ImportError, not an opaque crash."""
    # Arrange
    _ = seizure_metrics_absent
    module_name = "scitex_ml.metrics.seizure"
    # Act
    do_import = lambda: importlib.import_module(module_name)
    # Assert
    with pytest.raises(ImportError):
        do_import()


def test_seizure_shim_importerror_names_the_extra(seizure_metrics_absent):
    """The degraded ImportError points the user at the [seizure] install extra."""
    # Arrange
    _ = seizure_metrics_absent
    # Act
    try:
        importlib.import_module("scitex_ml.metrics.seizure")
        message = ""
    except ImportError as e:
        message = str(e)
    # Assert
    assert "scitex-ml[seizure]" in message


def test_seizure_shim_importerror_names_the_standalone(seizure_metrics_absent):
    """The degraded ImportError names the standalone package responsible."""
    # Arrange
    _ = seizure_metrics_absent
    # Act
    try:
        importlib.import_module("scitex_ml.metrics.seizure")
        message = ""
    except ImportError as e:
        message = str(e)
    # Assert
    assert "scitex-seizure-metrics" in message


def test_non_seizure_metric_still_imports_when_absent(seizure_metrics_absent):
    """A non-optional metric (calc_bacc) is unaffected by the absence."""
    # Arrange
    _ = seizure_metrics_absent
    # Act
    metrics = importlib.import_module("scitex_ml.metrics")
    # Assert
    assert hasattr(metrics, "calc_bacc")


def test_pure_numpy_seizure_metric_still_runs_when_absent(seizure_metrics_absent):
    """The pure-numpy window metric (no standalone needed) still computes."""
    # Arrange
    import numpy as np
    import pandas as pd

    metrics = importlib.import_module("scitex_ml.metrics")
    y_true = np.array(["seizure", "seizure", "interictal_control"])
    y_pred = np.array(["seizure", "interictal_control", "interictal_control"])
    metadata = pd.DataFrame(
        {"seizure_type": ["seizure", "seizure", "interictal_control"]}
    )
    # Act
    result = metrics.calc_seizure_window_prediction_metrics(y_true, y_pred, metadata)
    # Assert
    assert result["seizure_sensitivity"] == 50.0
