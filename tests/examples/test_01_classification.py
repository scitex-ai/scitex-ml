"""Smoke test for examples/01_classification.ipynb via jupyter nbconvert --execute.

Per PS505 (SciTeX audit): notebook smoke tests must invoke
``jupyter nbconvert --execute`` or ``pytest --nbval[-lax]``. nbconvert
is the canonical SciTeX choice.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbconvert")

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "01_classification.ipynb"
_NB_EXISTS = NOTEBOOK.is_file()


@pytest.mark.skipif(not _NB_EXISTS, reason=f"missing notebook: {NOTEBOOK}")
def test_classification_notebook_executes_with_nbconvert_and_exits_zero(tmp_path):
    # Arrange
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    cmd = [
        # `-m nbconvert` (not `-m jupyter nbconvert`): the latter goes
        # through jupyter_core's subcommand dispatch, which resolves
        # `jupyter-nbconvert` by searching PATH — NOT via sys.path — so
        # on a machine with a stray global `~/.local/bin/jupyter-nbconvert`
        # ahead of this venv on PATH, it launches that Python instead of
        # this venv's, crashing with ModuleNotFoundError: nbconvert.
        # `-m nbconvert` uses Python's own sys.path-based module
        # resolution, always the interpreter actually running this test.
        sys.executable, "-m", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=180", str(target),
    ]
    # Act
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    # Assert
    assert proc.returncode == 0
