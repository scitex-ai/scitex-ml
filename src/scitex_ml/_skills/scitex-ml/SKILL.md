---
name: scitex-ml
description: Classical and deep machine-learning utilities — classification reporters, time-series CV, training helpers, loss/metrics/optim, clustering, sampling, sklearn integration. Drop-in for sklearn workflows that need reproducible reporting.
primary_interface: python
interfaces: {python: 3, cli: 0, mcp: 0, skills: 2, hook: 0, http: 0}
---

# scitex-ml

Classical / deep ML utilities for scientific research. Standalone home of
what used to live in `scitex.ai` (the umbrella now exposes this package as
`scitex.ml`). Generative-AI providers were factored out to
[scitex-genai](https://github.com/ywatanabe1989/scitex-genai).

## Sub-skills

* [classification.md](classification.md) — Classifier, ClassificationReporter, time-series CV
* [training.md](training.md) — EarlyStopping, LearningCurveLogger
* [loss.md](loss.md) — MultiTaskLoss, L1/L2/Elastic regularization
* [optim.md](optim.md) — get_optimizer, set_optimizer, Ranger support
* [clustering.md](clustering.md) — pca(), umap() dimensionality reduction
* [metrics.md](metrics.md) — calc_bacc, calc_conf_mat, calc_roc_auc, silhouette scores
* [sampling.md](sampling.md) — undersample() for imbalanced data
* [feature-selection.md](feature-selection.md) — feature importance, univariate selection

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
