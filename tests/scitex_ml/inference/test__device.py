#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_ml/inference/test__device.py
# ----------------------------------------
"""Tests for scitex_ml.inference._device (SciTeX Voice V1).

No mocks (PA-306): the env-override test manipulates the real process
environment and restores it in a fixture.
"""

from __future__ import annotations

import os

import pytest

from scitex_ml.inference import PASCAL_CC, resolve_device

_ENV = "SCITEX_ML_DEVICE"


@pytest.fixture
def clean_device_env():
    """Save/restore the real SCITEX_ML_DEVICE around a test."""
    saved = os.environ.get(_ENV)
    os.environ.pop(_ENV, None)
    yield os.environ
    if saved is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = saved


def test_explicit_prefer_argument_is_returned(clean_device_env):
    # Arrange
    clean_device_env[_ENV] = "cuda:1"

    # Act
    device = resolve_device("cpu")

    # Assert
    assert device == "cpu"


def test_env_override_selects_named_device(clean_device_env):
    # Arrange
    clean_device_env[_ENV] = "cuda:3"

    # Act
    device = resolve_device()

    # Assert
    assert device == "cuda:3"


def test_auto_resolves_to_cpu_without_cuda(clean_device_env):
    # Arrange
    _ = clean_device_env  # fixture already cleared SCITEX_ML_DEVICE

    # Act
    device = resolve_device()

    # Assert
    assert device in {"cpu", "cuda"}


def test_pascal_compute_capability_constant():
    # Arrange
    expected = (6, 1)

    # Act
    value = PASCAL_CC

    # Assert
    assert value == expected
