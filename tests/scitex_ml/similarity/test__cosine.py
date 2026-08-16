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
    # Arrange
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    # Act
    score = cosine_similarity(v, v)[0, 0]

    # Assert
    assert score == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_vectors_score_zero():
    # Arrange
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)

    # Act
    score = cosine_similarity(a, b)[0, 0]

    # Assert
    assert score == pytest.approx(0.0, abs=1e-6)


def test_opposite_vectors_score_minus_one():
    # Arrange
    a = np.array([1.0, 1.0], dtype=np.float32)

    # Act
    score = cosine_similarity(a, -a)[0, 0]

    # Assert
    assert score == pytest.approx(-1.0, abs=1e-5)


def test_magnitude_scaling_preserves_similarity():
    # Arrange
    a = np.array([1.0, 2.0, 2.0], dtype=np.float32)

    # Act
    score = cosine_similarity(a, 10.0 * a)[0, 0]

    # Assert
    assert score == pytest.approx(1.0, abs=1e-5)


def test_zero_vector_score_is_not_nan():
    # Arrange
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)

    # Act
    score = cosine_similarity(a, b)[0, 0]

    # Assert
    assert not np.isnan(score)


def test_zero_vector_score_degrades_to_zero():
    # Arrange
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)

    # Act
    score = cosine_similarity(a, b)[0, 0]

    # Assert
    assert score == pytest.approx(0.0)


def test_batch_inputs_return_matrix_shape():
    # Arrange
    a = np.random.RandomState(0).randn(3, 8).astype(np.float32)
    b = np.random.RandomState(1).randn(5, 8).astype(np.float32)

    # Act
    scores = cosine_similarity(a, b)

    # Assert
    assert scores.shape == (3, 5)


def test_dim_mismatch_raises_value_error():
    # Arrange
    a = np.ones(4)
    b = np.ones(5)

    # Act
    # Assert
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_to_profile_returns_python_float():
    # Arrange
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    profile = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    # Act
    score = cosine_similarity_to_profile(emb, profile)

    # Assert
    assert isinstance(score, float)


def test_to_profile_computes_expected_cosine():
    # Arrange
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    profile = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    # Act
    score = cosine_similarity_to_profile(emb, profile)

    # Assert
    assert score == pytest.approx(np.sqrt(0.5), abs=1e-5)


def test_to_profile_rejects_batch_input():
    # Arrange
    batch = np.ones((2, 3))
    profile = np.ones(3)

    # Act
    # Assert
    with pytest.raises(ValueError):
        cosine_similarity_to_profile(batch, profile)
