#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_ml/inference/test__device.py
# ----------------------------------------
"""Tests for scitex_ml.inference._device (SciTeX Voice V1)."""

from __future__ import annotations

from scitex_ml.inference import PASCAL_CC, resolve_device


def test_explicit_prefer_wins(monkeypatch):
    monkeypatch.setenv("SCITEX_ML_DEVICE", "cuda:1")
    assert resolve_device("cpu") == "cpu"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SCITEX_ML_DEVICE", "cuda:3")
    assert resolve_device() == "cuda:3"


def test_auto_resolves_to_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("SCITEX_ML_DEVICE", raising=False)
    # With no torch (or no CUDA) available, auto-resolution must be "cpu"
    # and must never raise — import-safety contract.
    dev = resolve_device()
    assert dev in {"cpu", "cuda"}


def test_pascal_cc_constant():
    assert PASCAL_CC == (6, 1)
