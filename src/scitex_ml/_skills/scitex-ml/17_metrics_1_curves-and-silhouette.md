---
description: |
  [TOPIC] Curve, silhouette, importance, and seizure-domain metrics
  [DETAILS] ROC-AUC and PR-AUC (binary or one-vs-rest multiclass), with optional FPR/TPR curve return. Block-based silhouette scores for large datasets plus exact variants. Tree/coef and permutation-based feature importance. Seizure-prediction domain metrics (window-, event-, and legacy aliases).
tags: [scitex-ml-metrics_1_curves-and-silhouette]
---

# Metrics — Curves, Silhouette, Importance, Seizure

Continuation of [10_metrics.md](16_metrics.md). All functions live in `scitex_ml.metrics` and return the same dict shape (`{"metric", "value", "fold", ...}`) when applicable.

## calc_roc_auc()

```python
def calc_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: Optional[List] = None,
    fold: Optional[int] = None,
    return_curve: bool = False,
) -> Dict[str, Any]
```

ROC AUC score. Handles binary and multiclass (OvR weighted average).

### Parameters
- `y_proba` — Probability array: shape `(n,)` for binary 1D, `(n, 2)` for binary 2-column, `(n, k)` for multiclass
- `return_curve` — Include FPR/TPR arrays in result (binary only)

### Return value
`{"metric": "roc_auc", "value": float, "fold": int}` and optionally `"curve": {"fpr": ..., "tpr": ..., "thresholds": ...}`

---

## calc_pre_rec_auc()

```python
def calc_pre_rec_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: Optional[List] = None,
    fold: Optional[int] = None,
) -> Dict[str, Any]
```

Precision-Recall AUC score.

---

## Silhouette Scores

```python
# Block-based (efficient for large datasets)
calc_silhouette_score_block(X, labels, block_size=1000)
calc_silhouette_samples_block(X, labels, block_size=1000)

# Exact (slow for large datasets)
calc_silhouette_score_slow(X, labels)
calc_silhouette_samples_slow(X, labels)
```

All return float (score) or array (samples).

---

## Feature Importance

```python
# From model attributes (tree or linear)
calc_feature_importance(
    model,
    feature_names: List[str],
    method: str = "auto",  # "auto", "tree", "coef"
) -> Optional[Dict[str, float]]

# Permutation-based
calc_permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
) -> Dict[str, float]
```

See [13_feature-selection.md](19_feature-selection.md) for the higher-level `extract_feature_importance()` helper.

---

## Seizure Prediction Metrics (Domain-specific)

```python
calc_seizure_window_prediction_metrics(y_true, y_pred, ...)
calc_seizure_event_prediction_metrics(y_true, y_pred, ...)
calc_seizure_prediction_metrics(...)  # Backward compat alias
```

Specialized metrics for event-based seizure prediction evaluation.

---

## Full import reference

```python
from scitex_ml.metrics import (
    calc_roc_auc,
    calc_pre_rec_auc,
    calc_silhouette_score_block,
    calc_silhouette_score_slow,
    calc_silhouette_samples_block,
    calc_silhouette_samples_slow,
    calc_feature_importance,
    calc_permutation_importance,
    calc_seizure_window_prediction_metrics,
    calc_seizure_event_prediction_metrics,
)
```
