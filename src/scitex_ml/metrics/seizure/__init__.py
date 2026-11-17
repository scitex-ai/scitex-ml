"""scitex_ml.metrics.seizure — re-export shim for scitex-seizure-metrics.

Implementation lives in the standalone ``scitex-seizure-metrics`` package
(PyPI: ``scitex-seizure-metrics``, import: ``scitex_seizure_metrics``).
Install it with the ``[all]`` extra:

    pip install "scitex-ml[all]"

This shim makes ``scitex_ml.metrics.seizure.detection`` and friends
resolve to the scitex_seizure_metrics submodules so users don't have
to remember the standalone import name.

If the standalone is not installed, importing this module raises a
clear ``ImportError`` pointing the user at the install instruction.
"""

from __future__ import annotations

import sys

from scitex_ml._optional import missing_optional_dependency as _missing_optional

try:
    import scitex_seizure_metrics as _impl

    # Mirror the public surface of scitex_seizure_metrics — submodules,
    # data classes, and the version string. Anything added there shows up
    # here automatically because we alias module objects via sys.modules.
    from scitex_seizure_metrics import (  # noqa: F401
        AlarmPolicy,
        MetricsReport,
        __version__,
        adapters,
        bridge,
        calibration,
        detection,
        forecasting,
        papers,
        plots,
        report,
        surrogates,
    )
except ImportError as _exc:
    raise _missing_optional(
        "scitex_ml.metrics.seizure", _exc, distribution="scitex-seizure-metrics"
    ) from _exc

# Make `scitex_ml.metrics.seizure.detection` resolve to the *same*
# module object as `scitex_seizure_metrics.detection`. Same for the
# other subpackages — any future deep imports stay consistent.
for _sub in (
    "adapters",
    "bridge",
    "calibration",
    "detection",
    "forecasting",
    "papers",
    "plots",
    "report",
    "surrogates",
):
    sys.modules[f"{__name__}.{_sub}"] = getattr(_impl, _sub)

__all__ = [
    "__version__",
    "adapters",
    "bridge",
    "calibration",
    "detection",
    "forecasting",
    "papers",
    "plots",
    "report",
    "surrogates",
    "AlarmPolicy",
    "MetricsReport",
]
