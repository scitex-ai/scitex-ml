---
name: scitex-ml
description: |
  [WHAT] Classical and deep machine-learning utilities for scientific
  workflows — Classifier factory over scikit-learn, ClassificationReporter
  for metric tracking + figures, time-series cross-validation splitters,
  EarlyStopping, LearningCurveLogger, MultiTaskLoss with uncertainty
  weighting, L1/L2/elastic regularizers, optimizer helpers (get_optimizer,
  set_optimizer, Ranger), pca/umap dimensionality reduction with
  multi-dataset subplots, metrics (balanced accuracy, MCC, confusion
  matrix, ROC-AUC, PR-AUC, silhouette, feature importance),
  undersampling, and feature-selection utilities.
  [WHEN] running classification or regression experiments with
  scikit-learn / PyTorch and you want reproducible metric reporting,
  CV pipelines, multi-task loss balancing, early stopping with
  checkpoint saving, or one-call PCA/UMAP visualisation. Trigger
  phrases: "ClassificationReporter", "balanced accuracy", "ROC AUC",
  "confusion matrix figure", "time-series CV", "MultiTaskLoss",
  "EarlyStopping", "set_optimizer", "Ranger optimizer", "feature
  importance", "PCA plot", "UMAP plot", "undersample".
  [HOW] `import scitex_ml` (standalone) or `import scitex` then
  `scitex.ml.Classifier(...)`, `scitex.ml.ClassificationReporter(...)`,
  `scitex.ml.set_optimizer(model, "adam", lr=1e-3)`,
  `scitex.ml.metrics.calc_bacc(y_true, y_pred)`. Sub-skills under
  this directory cover each surface in detail.
tags: [scitex-ml]
primary_interface: python
interfaces:
  python: 3
  cli: 0
  mcp: 0
  skills: 2
  hook: 0
  http: 0
---

# scitex-ml

Classical / deep ML utilities for scientific research. Standalone home of
what used to live in `scitex.ai` (the umbrella now exposes this package as
`scitex.ml`). Generative-AI providers were factored out to
[scitex-genai](https://github.com/ywatanabe1989/scitex-genai).

## Sub-skills

### Onboarding (canonical 01-03)
* [01_installation.md](01_installation.md) — pip install + extras + verify
* [02_quick-start.md](02_quick-start.md) — first classifier + reporter round-trip
* [03_python-api.md](03_python-api.md) — full Python surface map

### Core surfaces
* [04_classification.md](04_classification.md) — Classifier factory + ClassificationReporter
* [05_classification_1_cv-experiment.md](05_classification_1_cv-experiment.md) — CrossValidationExperiment + time-series CV
* [06_training.md](06_training.md) — EarlyStopping, LearningCurveLogger
* [07_loss.md](07_loss.md) — MultiTaskLoss, L1/L2/elastic regularizers
* [08_optim.md](08_optim.md) — get_optimizer, set_optimizer, Ranger
* [09_clustering.md](09_clustering.md) — pca(), umap() dimensionality reduction
* [10_metrics.md](10_metrics.md) — calc_bacc, calc_mcc, calc_conf_mat, calc_clf_report
* [11_metrics_1_curves-and-silhouette.md](11_metrics_1_curves-and-silhouette.md) — ROC-AUC, PR-AUC, silhouette, feature importance, seizure metrics
* [12_sampling.md](12_sampling.md) — undersample() for imbalanced data
* [13_feature-selection.md](13_feature-selection.md) — feature importance, univariate selection, CV consistency

## Quick Reference

```python
import scitex_ml

# Classification
clf_server = scitex_ml.Classifier(class_weight={0: 1.0, 1: 2.0})
clf = clf_server("SVC")
reporter = scitex_ml.ClassificationReporter("./results")
reporter.calculate_metrics(y_true, y_pred, y_proba)

# Training
early_stopping = scitex_ml.EarlyStopping(patience=10, direction="minimize")
logger = scitex_ml.LearningCurveLogger()

# Loss
mtl = scitex_ml.MultiTaskLoss(are_regression=[False, False])

# Optimizer
optimizer = scitex_ml.set_optimizer(model, "adam", lr=1e-3)

# Metrics
result = scitex_ml.metrics.calc_bacc(y_true, y_pred)
cm = scitex_ml.metrics.calc_conf_mat(y_true, y_pred)

# Time-series CV
from scitex_ml.classification import TimeSeriesStratifiedSplit
splitter = TimeSeriesStratifiedSplit(n_splits=5)
```

## Umbrella access

```python
import scitex
scitex.ml.Classifier  # same object as scitex_ml.Classifier
```
