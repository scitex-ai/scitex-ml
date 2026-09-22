"""Isolation for the e2e layer: headless matplotlib + temp SCITEX_DIR."""

from __future__ import annotations

import os

import pytest

# Read before any test module imports scitex_ml (which pulls in
# matplotlib): force the headless backend so figure creation cannot
# block on a display that CI runners do not have.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _isolated_scitex_dir(tmp_path):
    """Redirect SCITEX_DIR at a per-test tmp dir (explicit save/restore)."""
    previous = os.environ.get("SCITEX_DIR")
    os.environ["SCITEX_DIR"] = str(tmp_path / ".scitex")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SCITEX_DIR", None)
        else:
            os.environ["SCITEX_DIR"] = previous
