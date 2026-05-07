<!-- ---
!-- Timestamp: 2026-05-05
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-ml/README.md
!-- --- -->

# SciTeX ML (`scitex-ml`)

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center">
  <a href="https://scitex-ml.readthedocs.io/">Full Documentation</a> · <code>pip install scitex-ml</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-ml/"><img src="https://img.shields.io/pypi/v/scitex-ml.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scitex-ml/"><img src="https://img.shields.io/pypi/pyversions/scitex-ml.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/scitex-ml/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/scitex-ml/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/scitex-ml"><img src="https://codecov.io/gh/ywatanabe1989/scitex-ml/graph/badge.svg" alt="Coverage"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Overview

`scitex-ml` is the standalone home of the classical/deep-ML utilities that
previously lived inside `scitex.ai` in the [`scitex-python`](https://github.com/ywatanabe1989/scitex-python)
umbrella. The generative-AI side (provider abstraction, agents) now lives
in [`scitex-genai`](https://github.com/ywatanabe1989/scitex-genai). This
package contains:

- **Classification**: `ClassificationReporter` (cross-validation aware
  reporting), time-series cross-validation splitters
  (`TimeSeriesStratifiedSplit`, `TimeSeriesBlockingSplit`,
  `TimeSeriesSlidingWindowSplit`, `TimeSeriesCalendarSplit`), `Classifier`
  pipeline glue.
- **Training**: `EarlyStopping`, `LearningCurveLogger`.
- **Loss / Metrics**: `MultiTaskLoss`, balanced accuracy, ROC/PR-curve
  helpers, seizure-prediction metrics.
- **Optim**: thin `get_optimizer` / `set_optimizer` shortcuts plus the
  vendored Ranger optimizer.
- **Clustering**: PCA + UMAP wrappers.
- **Feature extraction & selection**: ViT embeddings, univariate /
  multivariate feature selection, importance aggregation.
- **Plotting**: ROC / PR / learning-curve / confusion-matrix /
  feature-importance plots that integrate with the SciTeX cascade.
- **Sampling**: undersampling helpers.
- **Sklearn / sk**: scikit-learn integration helpers.

The umbrella `scitex.ai` namespace continues to work — it now thin-re-exports
this standalone (see
[re-export skill](https://github.com/ywatanabe1989/scitex-python/blob/main/_skills/general/01_ecosystem_05_re-export.md)),
so `scitex.ai.X` and `scitex_ml.X` resolve to the same object.

## Installation

```bash
pip install scitex-ml          # core
pip install scitex-ml[heavy]   # + torch / catboost / optuna / psutil
pip install scitex-ml[mcp]     # + fastmcp
pip install scitex-ml[all]     # everything
```

## Python API ⭐⭐⭐

`scitex-ml` ships a single Python API surface — `import scitex_ml` (or
`scitex.ml` via the umbrella). It has no console-script CLI and no MCP
server of its own; ML workflows are composed in Python and run via the
umbrella `scitex` CLI / session decorator.

```python
from scitex_ml import Classifier, EarlyStopping, ClassificationReporter
```

## CLI ⭐ — none

This package has no dedicated CLI. Use the umbrella `scitex` CLI
(`scitex session`, `scitex dev`, …) to drive ML workflows from the
shell.

## MCP ⭐ — none

No MCP server.

## Skills ⭐⭐

Skill index lives at `src/scitex_ml/_skills/scitex-ml/SKILL.md`. Loaded
automatically by Claude when working on this package.

## Quick start

### Training utilities

```python
from scitex_ml import EarlyStopping, LearningCurveLogger

stopper = EarlyStopping(patience=10, delta=1e-3)
logger = LearningCurveLogger(log_dir="./logs")
```

### Classification reporting

```python
from scitex_ml import ClassificationReporter

reporter = ClassificationReporter(save_dir="./results")
reporter.calc_metrics(y_true, y_pred, y_prob, labels=["pos", "neg"])
reporter.summarize()
reporter.save()
```

### Time-series CV

```python
from scitex_ml.classification import (
    TimeSeriesStratifiedSplit,
    TimeSeriesSlidingWindowSplit,
)

splitter = TimeSeriesStratifiedSplit(n_splits=5)
for train_idx, val_idx in splitter.split(X, y, timestamps):
    ...
```

## Architecture

`scitex-ml` is a downstream/middle SciTeX package. It depends on:

| Standalone | Why |
|---|---|
| `scitex-logging` | ecosystem-wide `getLogger` |
| `scitex-io` | unified save/load (used by reporters and `_BaseGenAI` config loader) |
| `scitex-plt` | `scitex_plt.color` palette helpers |
| `scitex-repro` | `fix_seeds` for `MultiTaskLoss` |
| `scitex-types` | `ArrayLike` |

It does **not** import the umbrella `scitex` package at runtime; that would
form a cycle (the umbrella re-exports this package). See
[`02_package_02_project-structure-src.md`](https://github.com/ywatanabe1989/scitex-python/blob/main/_skills/general/02_package_02_project-structure-src.md).

## Origin

This package was factored out of `scitex.ai` in the umbrella
[`scitex-python`](https://github.com/ywatanabe1989/scitex-python) on
2026-05-05 (branch `feat/factor-out-from-umbrella`). The umbrella now
ships a thin re-export bridge under `scitex/ai/__init__.py`.

## License

AGPL-3.0-only. See [LICENSE](LICENSE). For commercial / institutional
licensing, see [CLA.md](CLA.md).

## Part of SciTeX

`scitex-ml` is part of [**SciTeX**](https://scitex.ai). Install via
the umbrella with `pip install scitex[ai]` to use as
`scitex.ai` (Python).

```python
import scitex

@scitex.session
def main(CONFIG=scitex.INJECTED):
    from scitex.ai import Classifier
    clf = Classifier("LogisticRegression")
    return 0
```

`scitex.ai` delegates to `scitex_ml` — they share the same API and registry.

The SciTeX system follows the Four Freedoms for Research below, inspired by [the Free Software Definition](https://www.gnu.org/philosophy/free-sw.en.html):

>Four Freedoms for Research
>
>0. The freedom to **run** your research anywhere — your machine, your terms.
>1. The freedom to **study** how every step works — from raw data to final manuscript.
>2. The freedom to **redistribute** your workflows, not just your papers.
>3. The freedom to **modify** any module and share improvements with the community.
>
>AGPL-3.0 — because we believe research infrastructure deserves the same freedoms as the software it runs on.

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
</p>

<!-- EOF -->
