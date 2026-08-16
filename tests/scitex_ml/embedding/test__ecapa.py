#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_ml/embedding/test__ecapa.py
# ----------------------------------------
"""Tests for scitex_ml.embedding._ecapa (SciTeX Voice V1).

These exercise the import-safety contract: the module and constructor must
work with no torch/speechbrain, and only surface a helpful install hint at
load() time.
"""

from __future__ import annotations

import importlib.util

import pytest

from scitex_ml.embedding import EMBEDDING_DIM, ECAPAEmbedder

_HAS_SPEECHBRAIN = importlib.util.find_spec("speechbrain") is not None


def test_embedding_dim_constant():
    assert EMBEDDING_DIM == 192


def test_construct_is_cheap_without_deps():
    # Constructing must not import torch/speechbrain or hit the network.
    emb = ECAPAEmbedder(device="cpu")
    assert emb.device == "cpu"
    assert emb._model is None


def test_non_16k_rejected_before_load():
    emb = ECAPAEmbedder(device="cpu")
    with pytest.raises(ValueError, match="16 kHz"):
        emb.embed([0.0, 0.1, 0.2], sample_rate=8000)


@pytest.mark.skipif(
    _HAS_SPEECHBRAIN, reason="speechbrain installed; hint path not exercised"
)
def test_load_without_speechbrain_raises_install_hint():
    emb = ECAPAEmbedder(device="cpu")
    with pytest.raises(RuntimeError, match=r"scitex-ml\[voice\]"):
        emb.load()
