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
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=180", str(target),
    ]
    # Act
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    # Assert
    assert proc.returncode == 0
