"""scitex_ml.metrics.seizure — re-export shim for scitex-seizure-metrics.

Implementation lives in the standalone ``scitex-seizure-metrics`` package
(PyPI: ``scitex-seizure-metrics``, import: ``scitex_seizure_metrics``).
Install via the ``[seizure]`` extra:

    pip install scitex-ml[seizure]

This shim makes ``scitex_ml.metrics.seizure.detection`` and friends
resolve to the scitex_seizure_metrics submodules so users don't have
to remember the standalone import name.

If the standalone is not installed, importing this module raises a
clear ``ImportError`` pointing the user at the install instruction.
"""

from __future__ import annotations

import sys

try:
    import scitex_seizure_metrics as _impl
    from scitex_seizure_metrics import (  # noqa: E402,F401
        AlarmPolicy,
        MetricsReport,
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
except ImportError as e:
    raise ImportError(
        "scitex_ml.metrics.seizure requires the scitex-seizure-metrics "
        "standalone package. Install with: pip install scitex-ml[seizure]"
    ) from e

__version__ = _impl.__version__

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
