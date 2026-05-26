---
description: |
  [TOPIC] Core classification metrics
  [DETAILS] Balanced accuracy, MCC, confusion matrix (with `pd.DataFrame` value), classification report, and `calc_bacc_from_conf_mat` shortcut. All `calc_*` helpers return dicts with `metric`/`value`/`fold`/`labels` for ClassificationReporter compatibility.
tags: [scitex-ml-metrics]
---

# Metrics — Core

All functions in `scitex_ml.metrics` return dictionaries containing `"metric"`, `"value"`, `"fold"`, and optionally `"labels"` or `"error"`.

See [11_metrics_1_curves-and-silhouette.md](11_metrics_1_curves-and-silhouette.md) for ROC/PR curves, silhouette, feature importance, and seizure-domain metrics.

## calc_bacc()

```python
def calc_bacc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    fold: Optional[int] = None,
) -> Dict[str, Any]
```

Balanced accuracy (average recall across classes).

### Return value
`{"metric": "balanced_accuracy", "value": float, "fold": int, "labels": list}`

### Example
```python
import scitex

result = scitex.ml.metrics.calc_bacc(y_true, y_pred)
print(f"Balanced accuracy: {result['value']:.3f}")
```

---

## calc_mcc()

```python
def calc_mcc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    fold: Optional[int] = None,
) -> Dict[str, Any]
```

Matthews Correlation Coefficient — ranges from -1 to +1.

---

## calc_conf_mat()

```python
def calc_conf_mat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    fold: Optional[int] = None,
    normalize: Optional[str] = None,
) -> Dict[str, Any]
```

### Parameters
- `normalize` — `"true"` (row-normalize), `"pred"` (column-normalize), `"all"` (total), or `None`

### Return value
`{"metric": "confusion_matrix", "value": pd.DataFrame, "fold": int, "labels": list, "normalize": ...}`

The `"value"` is a `pd.DataFrame` with class labels as both index and columns.

### Example
```python
result = scitex.ml.metrics.calc_conf_mat(
    y_true, y_pred,
    labels=["Cat", "Dog", "Bird"],
    normalize="true",
)
print(result["value"])  # pd.DataFrame
```

---

## calc_clf_report()

```python
def calc_clf_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    fold: Optional[int] = None,
) -> Dict[str, Any]
```

Wraps `sklearn.metrics.classification_report`. The `"value"` is the formatted report string.

---

## calc_bacc_from_conf_mat()

```python
def calc_bacc_from_conf_mat(
    conf_mat: np.ndarray,
) -> float
```

Computes balanced accuracy directly from a confusion matrix array.

---

## Core import reference

```python
from scitex_ml.metrics import (
    calc_bacc,
    calc_mcc,
    calc_conf_mat,
    calc_clf_report,
    calc_bacc_from_conf_mat,
)
```
