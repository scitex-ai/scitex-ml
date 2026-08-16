#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/similarity/_cosine.py
# ----------------------------------------
from __future__ import annotations

"""Cosine similarity between embeddings.

Part of the SciTeX Voice V1 seam: scitex-ml returns *numbers* (cosine
scores); scitex-audio owns the admit/discard *decision* and the tunable
threshold. There is deliberately no threshold constant in this file — the
gate value is measured from a ROC curve on real audio and stored per
speaker profile (see the V1 measurement plan on card
`scitex-voice-speaker-verified-dictation-v1-20260816`).
"""

from typing import Union

import numpy as np

__all__ = [
    "cosine_similarity",
    "cosine_similarity_to_profile",
]

ArrayLike = Union[np.ndarray, "list[float]"]


def _as_2d(x: ArrayLike) -> np.ndarray:
    """Coerce a 1-D vector or 2-D batch to a float32 2-D array (n, d)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {arr.shape}")
    return arr


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Cosine similarity between rows of ``a`` and rows of ``b``.

    Accepts 1-D vectors or 2-D batches. Returns an ``(n_a, n_b)`` matrix of
    cosine scores in ``[-1, 1]``. Zero-norm rows yield ``0.0`` rather than
    NaN so a silent/empty frame degrades to "not a match".

    Args:
        a: Embedding or batch of embeddings, shape ``(d,)`` or ``(n_a, d)``.
        b: Embedding or batch of embeddings, shape ``(d,)`` or ``(n_b, d)``.

    Returns:
        ``np.ndarray`` of shape ``(n_a, n_b)``, dtype float32.
    """
    A = _as_2d(a)
    B = _as_2d(b)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"embedding dims differ: {A.shape[1]} vs {B.shape[1]}"
        )
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    A_safe = np.divide(A, A_norm, out=np.zeros_like(A), where=A_norm != 0)
    B_safe = np.divide(B, B_norm, out=np.zeros_like(B), where=B_norm != 0)
    return (A_safe @ B_safe.T).astype(np.float32)


def cosine_similarity_to_profile(
    embedding: ArrayLike, profile_mean: ArrayLike
) -> float:
    """Cosine of one embedding against an enrolled profile mean vector.

    Convenience wrapper returning a single Python float — the scalar the
    scitex-audio verify gate compares against its (measured) threshold.

    Args:
        embedding: A single embedding, shape ``(d,)``.
        profile_mean: The enrolled speaker's mean embedding, shape ``(d,)``.

    Returns:
        Cosine similarity as a Python ``float``.
    """
    emb = _as_2d(embedding)
    if emb.shape[0] != 1:
        raise ValueError("cosine_similarity_to_profile expects a single embedding")
    return float(cosine_similarity(emb, profile_mean)[0, 0])


# EOF
