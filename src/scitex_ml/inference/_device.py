#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/inference/_device.py
# ----------------------------------------
from __future__ import annotations

"""Inference device / precision seam for SciTeX Voice V1.

This is the single place the CUDA-12.x-vs-CPU decision lives, so the
embedding model never hard-codes ``.cuda()``. See card
`scitex-voice-speaker-verified-dictation-v1-20260816`.

VERIFIED HARDWARE CONSTRAINT (compute-03, 2026-08-16):
  * GPU is a GTX 1070 — Pascal, compute capability 6.1.
  * CUDA Toolkit 13.0 dropped offline-compile/library support for
    Maxwell/Pascal/Volta; CUDA 12.x still builds for sm_61. Driver 580 is
    the last line to support GTX 1000, so on this box 12.x-vs-13 is a
    runtime choice, not a driver limit.
  * PyTorch 2.8+ dropped prebuilt Pascal (sm_60/61) wheels (pytorch
    #157517). If the embedding runs on GPU it must use a torch<=2.7 cu12x
    wheel; otherwise fall back to CPU. Embedding is cheap relative to ASR
    (one-machine-first, non-negotiable #6), so CPU may be sufficient — to
    be decided by the M0 microbench, not assumed here.
"""

import os
from typing import Optional

__all__ = ["resolve_device", "PASCAL_CC"]

# Compute capability of the GTX 1070 on compute-03/04.
PASCAL_CC = (6, 1)

# Env override so the M0 microbench can force cpu/cuda without code changes.
_ENV_DEVICE = "SCITEX_ML_DEVICE"


def resolve_device(prefer: Optional[str] = None) -> str:
    """Resolve the torch device string to run inference on.

    Resolution order:
      1. explicit ``prefer`` argument ("cpu" / "cuda" / "cuda:N"),
      2. ``$SCITEX_ML_DEVICE`` environment override,
      3. auto: "cuda" iff torch is importable AND reports CUDA available,
         else "cpu".

    torch is imported lazily so this module (and ``import scitex_ml``) works
    with no torch installed — it simply resolves to "cpu".

    Args:
        prefer: Optional explicit device string.

    Returns:
        A torch-compatible device string, e.g. "cpu" or "cuda:0".
    """
    choice = prefer or os.environ.get(_ENV_DEVICE)
    if choice:
        return choice

    try:
        import torch  # noqa: PLC0415 — lazy: keep import-time light & torch-optional

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# EOF
