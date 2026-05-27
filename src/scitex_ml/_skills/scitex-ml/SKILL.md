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
  cli: 2
  mcp: 2
  skills: 3
  hook: 0
  http: 0
---

# scitex-ml

Classical / deep ML utilities for scientific research. Standalone home of
what used to live in `scitex.ai` (the umbrella now exposes this package as
`scitex.ml`). Generative-AI providers were factored out to
[scitex-genai](https://github.com/ywatanabe1989/scitex-genai).

## Sub-skills

### Onboarding + interfaces (01-05)
* [01_installation.md](01_installation.md) — pip install + extras + verify
* [02_quick-start.md](02_quick-start.md) — first classifier + reporter round-trip
* [03_python-api.md](03_python-api.md) — full Python surface map
* [04_cli-reference.md](04_cli-reference.md) — `scitex-ml` CLI: analysis verbs, mcp/skills groups
* [05_mcp-tools.md](05_mcp-tools.md) — MCP tools (`ml_compute_metrics`, `ml_generate_report`, `ml_reduce_dimensions`)

### Core surfaces (10-19)
* [10_classification.md](10_classification.md) — Classifier factory + ClassificationReporter
* [11_classification_1_cv-experiment.md](11_classification_1_cv-experiment.md) — CrossValidationExperiment + time-series CV
* [12_training.md](12_training.md) — EarlyStopping, LearningCurveLogger
* [13_loss.md](13_loss.md) — MultiTaskLoss, L1/L2/elastic regularizers
* [14_optim.md](14_optim.md) — get_optimizer, set_optimizer, Ranger
* [15_clustering.md](15_clustering.md) — pca(), umap() dimensionality reduction
* [16_metrics.md](16_metrics.md) — calc_bacc, calc_mcc, calc_conf_mat, calc_clf_report
* [17_metrics_1_curves-and-silhouette.md](17_metrics_1_curves-and-silhouette.md) — ROC-AUC, PR-AUC, silhouette, feature importance
* [18_sampling.md](18_sampling.md) — undersample() for imbalanced data
* [19_feature-selection.md](19_feature-selection.md) — feature importance, univariate selection, CV consistency

## Quick Reference

```python
import scitex_ml

clf = scitex_ml.Classifier(class_weight={0: 1.0, 1: 2.0})("SVC")
reporter = scitex_ml.ClassificationReporter("./results")
reporter.calculate_metrics(y_true, y_pred, y_proba)
result = scitex_ml.metrics.calc_bacc(y_true, y_pred)
optimizer = scitex_ml.set_optimizer(model, "adam", lr=1e-3)
```

See the leaves above for training, loss, time-series CV, clustering and
feature-selection details.

## CLI + MCP (stateless analysis surface)

The CLI and MCP server expose the same three file-in → artifact-out verbs
(identical JSON — CLI ↔ MCP parity). Training, optimizers, EarlyStopping and
the deep submodule API stay Python-only by design.

```bash
scitex-ml compute-metrics preds.csv --json        # bacc / mcc / confusion / AUC
scitex-ml generate-report preds.csv -o ./report   # full report + plots
scitex-ml reduce-dimensions feats.csv -o pca.png --label-col target
scitex-ml mcp start                                # MCP server (stdio)
```

| CLI verb | MCP tool (umbrella-mounted) | Python API wrapped |
|---|---|---|
| `compute-metrics` | `ml_compute_metrics` | `scitex_ml.metrics.*` |
| `generate-report` | `ml_generate_report` | `scitex_ml.ClassificationReporter` |
| `reduce-dimensions` | `ml_reduce_dimensions` | `scitex_ml.clustering.pca/umap` |

**Parity note:** scitex-ml deliberately exposes only this stateless analysis
slice via CLI+MCP, so it sets `mcp_parity_exempt = true` (`[tool.scitex_dev]`)
— the strict §6 Python-API↔MCP 1:1 parity check is a false positive for a
package whose bulk API is stateful/in-process.

## Umbrella access

```python
import scitex
scitex.ml.Classifier  # same object as scitex_ml.Classifier
```
