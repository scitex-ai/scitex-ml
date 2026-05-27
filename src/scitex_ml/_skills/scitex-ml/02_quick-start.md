---
description: |
  [TOPIC] scitex-ml 60-second quick-start
  [DETAILS] First Classifier + ClassificationReporter round-trip with cross-validation, EarlyStopping for PyTorch loops, and one-line balanced-accuracy / confusion-matrix metric calls.
tags: [scitex-ml-quick-start]
---

# Quick Start

## Classifier + ClassificationReporter in one CV loop

```python
import numpy as np
import scitex_ml
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

X, y = make_classification(n_samples=400, n_features=20, random_state=0)

# Classifier factory — initialises any of ~10 sklearn classifiers with a
# consistent interface and an optional scaler pipeline.
clf_server = scitex_ml.Classifier(class_weight={0: 1.0, 1: 1.0})
clf = clf_server("SVC")

reporter = scitex_ml.ClassificationReporter("./results/binary")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr, te) in enumerate(cv.split(X, y)):
    clf.fit(X[tr], y[tr])
    y_pred = clf.predict(X[te])
    y_proba = clf.predict_proba(X[te]) if hasattr(clf, "predict_proba") else None
    reporter.calculate_metrics(
        y_true=y[te],
        y_pred=y_pred,
        y_proba=y_proba,
        labels=["Negative", "Positive"],
        fold=fold,
    )

reporter.save_summary()       # ./results/binary/summary.json + CV figures
```

## EarlyStopping for PyTorch training loops

```python
import scitex_ml

early_stopping = scitex_ml.EarlyStopping(
    patience=10, verbose=True, direction="minimize"
)

for i_global in range(max_iters):
    val_loss = evaluate(model)
    should_stop = early_stopping(
        current_score=val_loss,
        models_spaths_dict={model: "./checkpoints/best.pth"},
        i_global=i_global,
    )
    if should_stop:
        break
```

## One-call metrics

```python
import scitex_ml

bacc = scitex_ml.metrics.calc_bacc(y_true, y_pred)
cm   = scitex_ml.metrics.calc_conf_mat(y_true, y_pred, labels=["Neg", "Pos"])
auc  = scitex_ml.metrics.calc_roc_auc(y_true, y_proba)

print(f"Balanced accuracy: {bacc['value']:.3f}")
print(cm["value"])            # pd.DataFrame
```

## Umbrella access

If you have `scitex` installed alongside, the same surface is also
reachable as `scitex.ml`:

```python
import scitex
scitex.ml.Classifier  # same object as scitex_ml.Classifier
```

## Next steps

- [03_python-api.md](03_python-api.md) — full Python API surface map
- [10_classification.md](10_classification.md) — Classifier + ClassificationReporter deep dive
- [11_classification_1_cv-experiment.md](11_classification_1_cv-experiment.md) — CrossValidationExperiment, time-series CV
- [12_training.md](12_training.md) — EarlyStopping + LearningCurveLogger
- [16_metrics.md](16_metrics.md) — full metrics surface
