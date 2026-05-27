---
description: |
  [TOPIC] CrossValidationExperiment + time-series CV splitters
  [DETAILS] High-level CV runner that drives ClassificationReporter from a model-factory function and CV splitter, plus four time-series-aware splitters (Stratified, Blocking, SlidingWindow, Calendar) and the Strategy/Metadata helpers.
tags: [scitex-ml-classification_1_cv-experiment]
---

# CrossValidationExperiment & Time-Series CV

## CrossValidationExperiment

High-level helper that runs a full CV experiment from a model factory function.

```python
class CrossValidationExperiment:
    def __init__(
        self,
        name: str,
        model_fn: Callable,
        cv: Optional[BaseCrossValidator] = None,
        output_dir: Optional[Union[str, Path]] = None,
        metrics: Optional[List[str]] = None,
        save_models: bool = True,
        verbose: bool = True,
    )
```

### Parameters
- `name` — Experiment name (used in output paths)
- `model_fn` — Callable that returns a fresh model instance per fold
- `cv` — Splitter instance. Default: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `save_models` — Whether to pickle models per fold. Default: `True`

### run()

```python
def run(
    self,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    calculate_curves: bool = True,
) -> Dict[str, Any]
```

Returns dict with `"paths"`, `"metadata"`, `"timing"`, `"models"`.

### Example

```python
import scitex
from sklearn.svm import SVC

experiment = scitex.ml.CrossValidationExperiment(
    name="svm_binary",
    model_fn=lambda: SVC(probability=True),
    output_dir="./cv_results",
)
results = experiment.run(X, y, class_names=["Neg", "Pos"])
```

### quick_experiment()

Convenience function for rapid experimentation:

```python
from scitex_ml.classification import quick_experiment

results = quick_experiment(
    X, y, model=SVC(), name="quick_svm", n_folds=5
)
```

---

## Time-series CV splitters

Available in `scitex_ml.classification.timeseries`:

| Class | Description |
|---|---|
| `TimeSeriesStratifiedSplit` | Stratified split preserving class balance in time series |
| `TimeSeriesBlockingSplit` | Non-overlapping block CV for time series |
| `TimeSeriesSlidingWindowSplit` | Rolling / expanding window CV |
| `TimeSeriesCalendarSplit` | Split by calendar period (day/week/month) |
| `TimeSeriesStrategy` | Strategy pattern for flexible splitter selection |
| `TimeSeriesMetadata` | Metadata container for time series datasets |

### Example

```python
from scitex_ml.classification.timeseries import TimeSeriesSlidingWindowSplit

splitter = TimeSeriesSlidingWindowSplit(n_splits=5)
for train_idx, test_idx in splitter.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

### Pairing with ClassificationReporter

```python
import scitex

splitter = TimeSeriesSlidingWindowSplit(n_splits=5)
reporter = scitex.ml.ClassificationReporter("./results/ts")

for fold, (tr, te) in enumerate(splitter.split(X, y)):
    model.fit(X[tr], y[tr])
    reporter.calculate_metrics(
        y_true=y[te],
        y_pred=model.predict(X[te]),
        y_proba=model.predict_proba(X[te]),
        fold=fold,
    )
reporter.save_summary()
```

## See also

- [04_classification.md](10_classification.md) — Classifier + ClassificationReporter
- [06_training.md](12_training.md) — EarlyStopping, LearningCurveLogger
