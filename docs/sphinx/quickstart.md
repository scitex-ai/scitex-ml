# Quickstart

```python
import scitex_ml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, stratify=y)

# Classifier — thin factory over scikit-learn estimators.
clf = scitex_ml.Classifier("LogisticRegression")
clf.fit(X_tr, y_tr)
print(f"test accuracy: {clf.score(X_te, y_te):.3f}")

# ClassificationReporter — metric tracking + figure export.
reporter = scitex_ml.ClassificationReporter(output_dir="./results")
reporter.calculate_metrics(y_te, clf.predict(X_te), clf.predict_proba(X_te))
reporter.save_summary()
```

For a runnable walk-through see [`examples/01_classification.ipynb`](https://github.com/ywatanabe1989/scitex-ml/blob/main/examples/01_classification.ipynb).
