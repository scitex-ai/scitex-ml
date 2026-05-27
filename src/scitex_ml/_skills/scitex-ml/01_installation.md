---
description: |
  [TOPIC] Installing scitex-ml
  [DETAILS] pip install (standalone vs umbrella), optional ML / deep-learning extras (sklearn, torch, umap-learn, imbalanced-learn, pytorch-optimizer), and how to verify the install.
tags: [scitex-ml-installation]
---

# Installation

## pip install

```bash
pip install scitex-ml                # core: Classifier, ClassificationReporter, metrics
pip install 'scitex-ml[all]'         # everything (deep-learning extras)
```

Requires Python >= 3.9.

## Standalone vs umbrella

`scitex-ml` is a standalone package, but it is also part of the
[scitex umbrella](https://pypi.org/project/scitex/). The same module
is reachable via two import paths:

```python
# Standalone — pip install scitex-ml
import scitex_ml
clf_server = scitex_ml.Classifier()

# Umbrella — pip install scitex
import scitex
clf_server = scitex.ml.Classifier()
```

`pip install scitex-ml` alone does **not** expose the `scitex`
namespace; `import scitex.ml` raises `ModuleNotFoundError`. To get
both paths, install both: `pip install scitex scitex-ml` (or
`pip install scitex[ml]`).

## Optional extras

Heavy / format-specific dependencies are imported lazily via
`scitex_dev.try_import_optional`. A missing dep does not crash
`import scitex_ml`; the feature becomes unavailable at use-site and a
clear `ImportError` (with `pip install ...` hint) fires only when
called. Each gated dep exposes a `<NAME>_AVAILABLE` boolean for code
that wants to feature-flag without raising.

Install extras as needed:

```bash
pip install scikit-learn              # Classifier, metrics, cross-validation
pip install torch                     # EarlyStopping checkpoints, MultiTaskLoss, Ranger
pip install umap-learn                # scitex_ml.clustering.umap
pip install imbalanced-learn          # scitex_ml.sampling.undersample
pip install pytorch-optimizer         # Ranger21 optimizer
```

## Verify

```python
import scitex_ml
print(scitex_ml.__version__)
print(scitex_ml.Classifier().list)    # → list of supported classifier strings
```

## See also

- [02_quick-start.md](02_quick-start.md) — first classifier + reporter round-trip
- [03_python-api.md](03_python-api.md) — full Python surface map
- [10_classification.md](10_classification.md) — Classifier + ClassificationReporter
