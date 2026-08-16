#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/embedding/__init__.py
# ----------------------------------------
"""Speaker (and general) embedding models (SciTeX Voice V1).

``ECAPAEmbedder`` is import-safe without torch/speechbrain: constructing it
is cheap and the heavy deps are pulled lazily at ``load()``/``embed()``,
which raise a clear install hint if missing. ``EMBEDDING_DIM`` is a plain
constant, always available.
"""

from ._ecapa import EMBEDDING_DIM, ECAPAEmbedder

__all__ = [
    "ECAPAEmbedder",
    "EMBEDDING_DIM",
]
