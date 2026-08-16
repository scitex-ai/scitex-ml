#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/similarity/__init__.py
# ----------------------------------------
"""Similarity scoring between embeddings (SciTeX Voice V1).

Pure-numpy, no heavy deps: cosine scoring is the comparison half of the
speaker-verification seam. Kept light so ``import scitex_ml`` stays fast.
"""

from ._cosine import cosine_similarity, cosine_similarity_to_profile

__all__ = [
    "cosine_similarity",
    "cosine_similarity_to_profile",
]
