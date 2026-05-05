#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ai/__init__.py
# ----------------------------------------
from __future__ import annotations
"""SciTeX AI — machine learning, classification, GenAI, training utilities.

This package was factored out of `scitex.ai` (the in-umbrella module of
`scitex-python`). It is the single source of truth for the AI/ML utilities
in the SciTeX ecosystem; the umbrella `scitex.ai` thin-re-exports from here.

Public API:
    - ClassificationReporter, Classifier
    - EarlyStopping, LearningCurveLogger
    - MultiTaskLoss
    - GenAI                (lazy — heavy provider deps load only on access)
    - get_optimizer, set_optimizer

Submodules:
    activation, classification, clustering, feature_extraction,
    feature_selection, loss, metrics, optim, plt, sampling, sklearn,
    training, utils, _gen_ai
"""


import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("scitex-ai")
except _PackageNotFoundError:
    __version__ = "0.1.0+local"

# ---------------------------------------------------------------------------
# Eager submodule imports (cheap modules and ones used widely)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
from .classification import ClassificationReporter, Classifier
from .loss import MultiTaskLoss
from .optim import get_optimizer, set_optimizer
from .training._EarlyStopping import EarlyStopping
from .training._LearningCurveLogger import LearningCurveLogger


# ---------------------------------------------------------------------------
# Lazy GenAI (anthropic + openai + google-genai + groq are heavy)
# ---------------------------------------------------------------------------
def __getattr__(name):
    if name == "GenAI":
        from ._gen_ai import GenAI

        return GenAI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Classes
    "ClassificationReporter",
    "Classifier",
    "EarlyStopping",
    "LearningCurveLogger",
    "MultiTaskLoss",
    "GenAI",  # lazy
    # Functions
    "get_optimizer",
    "set_optimizer",
    # Submodules
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
