#!/usr/bin/env python3
"""Uniform error for an optional runtime dependency that is not installed.

scitex-ml keeps its heavy capabilities (torch, torchvision, optuna, ViT,
the MCP server, seizure metrics) out of ``[project.dependencies]``. Every
module that needs one guards the import and, when it is missing, raises the
error built here, so a user always sees the same single install line.
"""

from __future__ import annotations

INSTALL_HINT = 'pip install "scitex-ml[all]"'


def missing_optional_dependency(
    feature: str, exc: ImportError, distribution: str | None = None
) -> ImportError:
    """Return an ImportError naming the missing module and the install line.

    Args:
        feature: The scitex-ml capability that needs the dependency,
            e.g. ``"scitex_ml.loss"``.
        exc: The original ImportError raised by the guarded import.
        distribution: The PyPI distribution that provides the module, when
            its name differs from the import name (``scitex-seizure-metrics``
            provides ``scitex_seizure_metrics``).

    Returns:
        An ImportError to raise ``from exc`` so the original cause is kept.
    """
    missing = exc.name or str(exc)
    provider = f" (PyPI: {distribution})" if distribution else ""
    return ImportError(
        f"{feature} requires the optional dependency {missing!r}{provider}, "
        f"which is not installed. Install it with: {INSTALL_HINT}",
        name=exc.name,
    )


__all__ = ["INSTALL_HINT", "missing_optional_dependency"]
