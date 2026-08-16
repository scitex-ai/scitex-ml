#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/__init__.py
# ----------------------------------------
from __future__ import annotations

"""SciTeX ML — machine learning, classification, training utilities.

Factored out of the legacy `scitex.ai` module alongside `scitex-genai`
(which now owns the generative-AI provider abstraction). This package
is the single source of truth for classical/deep ML utilities in the
SciTeX ecosystem.

Public API:
    - ClassificationReporter, Classifier
    - EarlyStopping, LearningCurveLogger
    - MultiTaskLoss
    - get_optimizer, set_optimizer

Submodules:
    activation, classification, clustering, feature_extraction,
    feature_selection, loss, metrics, optim, plt, sampling, sk,
    sklearn, training, utils
"""


import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("scitex-ml")
except _PackageNotFoundError:
    __version__ = "0.0.0+local"

# ---------------------------------------------------------------------------
# Lazy import surface (PEP 562). `import scitex_ml` must stay fast — Click runs
# the CLI once per Tab press, so eagerly importing sklearn / matplotlib / sktime
# / umap here (≈6-8 s) would make tab-completion unusable (audit §10). Each
# public name is resolved on first access via `__getattr__` below and cached.
#
# Light submodules import directly; heavy ones go through `try_import_optional`
# so they resolve to `None` (not ImportError) in a no-`[heavy]` environment —
# downstream code probes with `is None` and surfaces an installable hint via
# `scitex_dev.last_install_hint` (Pattern A from
# `_skills/general/03_interface_01_python-api/04_lazy-imports-and-optional-deps.md`).
# ---------------------------------------------------------------------------
# Dispatch tables keyed by public name (PA-102 recognises dict-literal keys
# referenced via subscript in __getattr__, so these double as the bound-name
# registry). Light submodules -> relative module; heavy ones go through
# try_import_optional so a missing `[heavy]` dep resolves to None, not ImportError.
_LIGHT_SUBMODULES: dict[str, str] = {
    "classification": ".classification",
    "clustering": ".clustering",
    "embedding": ".embedding",
    "feature_extraction": ".feature_extraction",
    "feature_selection": ".feature_selection",
    "inference": ".inference",
    "metrics": ".metrics",
    "similarity": ".similarity",
    "plt": ".plt",
    "sampling": ".sampling",
    "sk": ".sk",
    "sklearn": ".sklearn",
    "utils": ".utils",
}
_HEAVY_SUBMODULES: dict[str, str] = {
    "activation": ".activation",
    "loss": ".loss",
    "optim": ".optim",
    "training": ".training",
}
# Public attribute -> light submodule it lives in.
_LIGHT_ATTRS: dict[str, str] = {
    "ClassificationReporter": ".classification",
    "Classifier": ".classification",
}
# Public attribute -> (heavy submodule path, attribute name).
_HEAVY_ATTRS: dict[str, tuple[str, str]] = {
    "MultiTaskLoss": (".loss", "MultiTaskLoss"),
    "get_optimizer": (".optim", "get_optimizer"),
    "set_optimizer": (".optim", "set_optimizer"),
    "EarlyStopping": (".training._EarlyStopping", "EarlyStopping"),
    "LearningCurveLogger": (".training._LearningCurveLogger", "LearningCurveLogger"),
}


def __getattr__(name: str):
    """PEP 562 lazy resolver for the public surface (see note above)."""
    from importlib import import_module

    if name in _LIGHT_SUBMODULES:
        value = import_module(_LIGHT_SUBMODULES[name], __name__)
    elif name in _LIGHT_ATTRS:
        value = getattr(import_module(_LIGHT_ATTRS[name], __name__), name)
    elif name in _HEAVY_SUBMODULES:
        from scitex_dev import try_import_optional

        value = try_import_optional(
            _HEAVY_SUBMODULES[name], extra="heavy", pkg="scitex-ml", package=__name__
        )
    elif name in _HEAVY_ATTRS:
        from scitex_dev import try_import_optional

        modpath, attr = _HEAVY_ATTRS[name]
        value = try_import_optional(
            modpath, attr=attr, extra="heavy", pkg="scitex-ml", package=__name__
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value  # cache so __getattr__ fires only once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__version__",
    "ClassificationReporter",
    "Classifier",
    "EarlyStopping",
    "LearningCurveLogger",
    "MultiTaskLoss",
    "get_optimizer",
    "set_optimizer",
    "activation",
    "classification",
    "clustering",
    "embedding",
    "feature_extraction",
    "inference",
    "loss",
    "metrics",
    "optim",
    "plt",
    "sampling",
    "similarity",
    "sklearn",
    "training",
    "utils",
]
