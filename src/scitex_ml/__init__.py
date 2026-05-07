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

from . import (
    activation,
    classification,
    clustering,
    feature_extraction,
    loss,
    metrics,
    optim,
    plt,
    sampling,
    sklearn,
    training,
    utils,
)
from .classification import ClassificationReporter, Classifier
from .loss import MultiTaskLoss
from .optim import get_optimizer, set_optimizer
from .training._EarlyStopping import EarlyStopping
from .training._LearningCurveLogger import LearningCurveLogger

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
