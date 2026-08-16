#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_ml/similarity/test__cosine.py
# ----------------------------------------
"""Tests for scitex_ml.similarity._cosine (SciTeX Voice V1)."""

from __future__ import annotations

import numpy as np
import pytest

from scitex_ml.similarity import cosine_similarity, cosine_similarity_to_profile


def test_identical_vectors_score_one():
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert cosine_similarity(v, v)[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_vectors_score_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_opposite_vectors_score_minus_one():
    a = np.array([1.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, -a)[0, 0] == pytest.approx(-1.0, abs=1e-5)


def test_magnitude_invariance():
    a = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    assert cosine_similarity(a, 10.0 * a)[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_zero_vector_degrades_to_zero_not_nan():
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    score = cosine_similarity(a, b)[0, 0]
    assert not np.isnan(score)
    assert score == pytest.approx(0.0)


def test_batch_shape():
    a = np.random.RandomState(0).randn(3, 8).astype(np.float32)
    b = np.random.RandomState(1).randn(5, 8).astype(np.float32)
    assert cosine_similarity(a, b).shape == (3, 5)


def test_dim_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(4), np.ones(5))


def test_to_profile_returns_float():
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    profile = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    score = cosine_similarity_to_profile(emb, profile)
    assert isinstance(score, float)
    assert score == pytest.approx(np.sqrt(0.5), abs=1e-5)


def test_to_profile_rejects_batch():
    with pytest.raises(ValueError):
        cosine_similarity_to_profile(np.ones((2, 3)), np.ones(3))
