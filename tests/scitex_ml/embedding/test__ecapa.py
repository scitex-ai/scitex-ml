#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_ml/embedding/test__ecapa.py
# ----------------------------------------
"""Tests for scitex_ml.embedding._ecapa (SciTeX Voice V1).

These exercise the import-safety contract: the module and constructor must
work with no torch/speechbrain installed, and the 16 kHz guard fires before
any heavy load. Loading the real model is covered by the M0 microbench on
compute-03, not here.
"""

from __future__ import annotations

import pytest

from scitex_ml.embedding import EMBEDDING_DIM, ECAPAEmbedder


def test_embedding_dim_is_192():
    # Arrange
    expected = 192

    # Act
    value = EMBEDDING_DIM

    # Assert
    assert value == expected


def test_construct_sets_requested_device():
    # Arrange
    device = "cpu"

    # Act
    embedder = ECAPAEmbedder(device=device)

    # Assert
    assert embedder.device == "cpu"


def test_construct_defers_model_load():
    # Arrange
    embedder = ECAPAEmbedder(device="cpu")

    # Act
    loaded = embedder._model

    # Assert
    assert loaded is None


def test_non_16k_sample_rate_is_rejected():
    # Arrange
    embedder = ECAPAEmbedder(device="cpu")

    # Act
    # Assert
    with pytest.raises(ValueError, match="16 kHz"):
        embedder.embed([0.0, 0.1, 0.2], sample_rate=8000)
