#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/__main__.py
# ----------------------------------------
"""Allow running the CLI as ``python -m scitex_ml``."""

import sys

from scitex_ml._cli import main

if __name__ == "__main__":
    sys.exit(main())


# EOF
