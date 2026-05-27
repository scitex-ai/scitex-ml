#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_cli/__init__.py
# ----------------------------------------
"""scitex-ml command-line interface — entry point ``scitex_ml._cli:main``."""

from ._main import main

__all__ = ["main"]

# audit §4 — root --help opens with the canonical `<cli> (vX.Y.Z) — …` line.
try:
    from importlib.metadata import version as _v

    main.help = f"scitex-ml (v{_v('scitex-ml')}) — " + (main.help or "").lstrip()
except Exception:  # pragma: no cover
    pass


# EOF
