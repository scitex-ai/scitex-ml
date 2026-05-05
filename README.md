<!-- ---
!-- Timestamp: 2026-05-05
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-ai/README.md
!-- --- -->

# SciTeX AI (`scitex-ai`)

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center">
  <a href="https://scitex-ai.readthedocs.io/">Full Documentation</a> · <code>pip install scitex-ai</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-ai/"><img src="https://img.shields.io/pypi/v/scitex-ai.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scitex-ai/"><img src="https://img.shields.io/pypi/pyversions/scitex-ai.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/scitex-ai/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/scitex-ai/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/scitex-ai"><img src="https://codecov.io/gh/ywatanabe1989/scitex-ai/graph/badge.svg" alt="Coverage"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Overview

`scitex-ai` is the standalone home of the AI/ML utilities that previously
lived inside `scitex.ai` in the [`scitex-python`](https://github.com/ywatanabe1989/scitex-python)
umbrella. It contains:

- **GenAI**: unified provider abstraction over OpenAI, Anthropic, Google,
  Groq, DeepSeek, Perplexity, and local Llama. Same API for every provider,
  cost tracking, conversation history, multi-modal where supported.
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
so `scitex.ai.X` and `scitex_ai.X` resolve to the same object.

## Install

```bash
pip install scitex-ai          # core
pip install scitex-ai[heavy]   # + torch / catboost / optuna / psutil
pip install scitex-ai[mcp]     # + fastmcp
pip install scitex-ai[all]     # everything
```

## Quick start

### GenAI

```python
from scitex_ai import GenAI

ai = GenAI(provider="openai", model="gpt-4o")
print(ai.complete("Explain neural networks in one sentence."))
print(ai.get_cost_summary())
```

### Training utilities

```python
from scitex_ai import EarlyStopping, LearningCurveLogger

stopper = EarlyStopping(patience=10, delta=1e-3)
logger = LearningCurveLogger(log_dir="./logs")
```

### Classification reporting

```python
from scitex_ai import ClassificationReporter

reporter = ClassificationReporter(save_dir="./results")
reporter.calc_metrics(y_true, y_pred, y_prob, labels=["pos", "neg"])
reporter.summarize()
reporter.save()
```

### Time-series CV

```python
from scitex_ai.classification import (
    TimeSeriesStratifiedSplit,
    TimeSeriesSlidingWindowSplit,
)

splitter = TimeSeriesStratifiedSplit(n_splits=5)
for train_idx, val_idx in splitter.split(X, y, timestamps):
    ...
```

## Architecture

`scitex-ai` is a downstream/middle SciTeX package. It depends on:

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
