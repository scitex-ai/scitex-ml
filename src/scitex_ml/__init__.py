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

from scitex_dev import try_import_optional

# ---------------------------------------------------------------------------
# Light submodules — these don't pull torch/optuna/catboost at import time.
# ---------------------------------------------------------------------------
from . import (
    classification,
    clustering,
    feature_extraction,
    metrics,
    plt,
    sampling,
    sklearn,
    utils,
)
from .classification import ClassificationReporter, Classifier

# ---------------------------------------------------------------------------
# Heavy submodules — gated via `try_import_optional` so that `import
# scitex_ml` succeeds in a no-`[heavy]` environment. Each entry resolves
# to `None` when its underlying dep (torch / optuna / catboost / …) is
# missing; downstream code probes with `is None` and surfaces an
# installable hint via `scitex_dev.last_install_hint`.
#
# Pattern A from
# `_skills/general/03_interface_01_python-api/04_lazy-imports-and-optional-deps.md`:
# the public names stay in `__all__` regardless.
# ---------------------------------------------------------------------------
activation = try_import_optional(
    ".activation", extra="heavy", pkg="scitex-ml", package=__name__
)
loss = try_import_optional(".loss", extra="heavy", pkg="scitex-ml", package=__name__)
optim = try_import_optional(".optim", extra="heavy", pkg="scitex-ml", package=__name__)
training = try_import_optional(
    ".training", extra="heavy", pkg="scitex-ml", package=__name__
)

MultiTaskLoss = try_import_optional(
    ".loss",
    attr="MultiTaskLoss",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)
get_optimizer = try_import_optional(
    ".optim",
    attr="get_optimizer",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)
set_optimizer = try_import_optional(
    ".optim",
    attr="set_optimizer",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)
EarlyStopping = try_import_optional(
    ".training._EarlyStopping",
    attr="EarlyStopping",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)
LearningCurveLogger = try_import_optional(
    ".training._LearningCurveLogger",
    attr="LearningCurveLogger",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)

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
    "feature_extraction",
    "loss",
    "metrics",
    "optim",
    "plt",
    "sampling",
    "sklearn",
    "training",
    "utils",
]
