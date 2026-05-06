#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal scitex-ai Classifier example.

Run:
    python examples/example_classifier.py
"""

from __future__ import annotations


def main() -> int:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    from scitex_ai import Classifier

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

    clf = Classifier("LogisticRegression")
    clf.fit(X_tr, y_tr)
    score = clf.score(X_te, y_te)
    print(f"test accuracy: {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
