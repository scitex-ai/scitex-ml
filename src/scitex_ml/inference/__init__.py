#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/inference/__init__.py
# ----------------------------------------
"""Inference runtime seam (SciTeX Voice V1).

Isolates the device/precision choice (CUDA 12.x on Pascal vs CPU) so model
code stays hardware-agnostic. Pure-python; torch is imported lazily inside
``resolve_device`` so this stays import-safe without torch.
"""

from ._device import PASCAL_CC, resolve_device

__all__ = [
    "PASCAL_CC",
    "resolve_device",
]
