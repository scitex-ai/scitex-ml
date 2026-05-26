---
description: |
  [TOPIC] scitex-ml Python API surface
  [DETAILS] Top-level public symbols (Classifier, ClassificationReporter, CrossValidationExperiment, EarlyStopping, LearningCurveLogger, MultiTaskLoss, set_optimizer, clustering.pca/umap, metrics, sampling, feature_selection) with one-line summaries and links to deep-dive sub-skills.
tags: [scitex-ml-python-api]
---

# Python API

All public symbols are importable from the top-level `scitex_ml`
package (or, via the umbrella, `scitex.ml`).

## Classification

```python
from scitex_ml import Classifier, ClassificationReporter
from scitex_ml.classification import CrossValidationExperiment, quick_experiment

clf_server = Classifier(class_weight={0: 1.0, 1: 2.0})
clf = clf_server("SVC")

reporter = ClassificationReporter("./results")
reporter.calculate_metrics(y_true, y_pred, y_proba, fold=0)
reporter.save_summary()
```

Deep dive: [04_classification.md](04_classification.md), [05_classification_1_cv-experiment.md](05_classification_1_cv-experiment.md).

## Time-series cross-validation

```python
from scitex_ml.classification import (
    TimeSeriesStratifiedSplit,
    TimeSeriesBlockingSplit,
    TimeSeriesSlidingWindowSplit,
    TimeSeriesCalendarSplit,
    TimeSeriesStrategy,
    TimeSeriesMetadata,
)
```

Deep dive: [05_classification_1_cv-experiment.md](05_classification_1_cv-experiment.md).

## Training helpers

```python
from scitex_ml import EarlyStopping, LearningCurveLogger
```

Deep dive: [06_training.md](06_training.md).

## Loss

```python
from scitex_ml import MultiTaskLoss
from scitex_ml.loss._L1L2Losses import l1, l2, elastic
```

Deep dive: [07_loss.md](07_loss.md).

## Optimizer

```python
from scitex_ml import set_optimizer, get_optimizer
```

Deep dive: [08_optim.md](08_optim.md).

## Clustering / dimensionality reduction

```python
from scitex_ml.clustering import pca, umap
```

Deep dive: [09_clustering.md](09_clustering.md).

## Metrics

```python
from scitex_ml.metrics import (
    calc_bacc, calc_mcc, calc_conf_mat, calc_clf_report,
    calc_roc_auc, calc_pre_rec_auc, calc_bacc_from_conf_mat,
    calc_silhouette_score_block, calc_silhouette_score_slow,
    calc_silhouette_samples_block, calc_silhouette_samples_slow,
    calc_feature_importance, calc_permutation_importance,
    calc_seizure_window_prediction_metrics,
    calc_seizure_event_prediction_metrics,
)
```

Deep dive: [10_metrics.md](10_metrics.md), [11_metrics_1_curves-and-silhouette.md](11_metrics_1_curves-and-silhouette.md).

## Sampling

```python
from scitex_ml.sampling import undersample
```

Deep dive: [12_sampling.md](12_sampling.md).

## Feature selection

```python
from scitex_ml.feature_selection import (
    extract_feature_importance,
    select_features_univariate,
    analyze_feature_consistency,
    aggregate_feature_importances,
    create_feature_importance_dataframe,
)
```

Deep dive: [13_feature-selection.md](13_feature-selection.md).

## Umbrella access

```python
import scitex
scitex.ml.Classifier                  # same object as scitex_ml.Classifier
scitex.ml.metrics.calc_bacc           # same object as scitex_ml.metrics.calc_bacc
```
