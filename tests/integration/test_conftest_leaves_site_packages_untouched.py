"""Regression test: importing tests/conftest.py must not touch site-packages."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"


def test_conftest_leaves_site_packages_untouched():
    # Arrange
    code = (
        "import os, runpy, site\n"
        "os.environ['COVERAGE_PROCESS_START'] = 'SENTINEL_PROCESS_START'\n"
        "os.environ['COVERAGE_FILE'] = 'SENTINEL_COVERAGE_FILE'\n"
        f"conftest = {str(CONFTEST_PATH)!r}\n"
        "dirs = []\n"
        "try:\n"
        "    dirs.extend(site.getsitepackages())\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    dirs.append(site.getusersitepackages())\n"
        "except Exception:\n"
        "    pass\n"
        "def snapshot():\n"
        "    snap = {}\n"
        "    for d in dirs:\n"
        "        try:\n"
        "            snap[d] = sorted(os.listdir(d))\n"
        "        except OSError:\n"
        "            snap[d] = []\n"
        "    return snap\n"
        "before = snapshot()\n"
        "runpy.run_path(conftest)\n"
        "after = snapshot()\n"
        "assert before == after, 'conftest.py modified site-packages'\n"
        "assert os.environ.get('COVERAGE_PROCESS_START') == 'SENTINEL_PROCESS_START'\n"
        "assert os.environ.get('COVERAGE_FILE') == 'SENTINEL_COVERAGE_FILE'\n"
    )
    # Act
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    # Assert
    assert proc.returncode == 0, proc.stderr + proc.stdout
