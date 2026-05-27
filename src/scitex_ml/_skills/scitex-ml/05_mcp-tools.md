---
description: |
  [TOPIC] scitex-ml MCP tools
  [DETAILS] Agent-callable MCP surface — ml_compute_metrics, ml_generate_report,
  ml_reduce_dimensions plus the mandatory ml_skills_list / ml_skills_get.
  Bare names in the standalone (compute_metrics, ...) surface as ml_* once the
  scitex umbrella mounts the server under namespace "ml".
tags: [scitex-ml-mcp-tools]
---

# MCP Tools

scitex-ml ships a single FastMCP server at `scitex_ml._mcp_server:mcp`. Tools
are defined with **bare** names; the scitex umbrella auto-mounts the server
under `namespace="ml"`, so agents call them as `ml_*`.

Start it with `scitex-ml mcp start` (needs `pip install scitex-ml[mcp]`), or
let the umbrella mount it automatically.

| Tool (umbrella) | Purpose | Wraps |
|---|---|---|
| `ml_compute_metrics` | Metrics from a saved predictions table (bacc, MCC, confusion matrix, ROC/PR-AUC) | `scitex_ml.metrics.*` |
| `ml_generate_report` | Full ClassificationReporter report (summary.json + plots) | `scitex_ml.ClassificationReporter` |
| `ml_reduce_dimensions` | PCA/UMAP 2-D projection figure | `scitex_ml.clustering.pca/umap` |
| `ml_skills_list` | List bundled skill pages | `_skills/scitex-ml/` |
| `ml_skills_get` | Fetch one skill page's Markdown | `_skills/scitex-ml/` |

All tools take/return JSON; the analysis tools return
`{"success": bool, ...}` and report input errors as
`{"success": false, "error": "..."}` rather than raising.

## Examples

```json
// ml_compute_metrics
{"predictions": "/data/preds.csv", "labels": ["neg", "pos"]}

// ml_generate_report
{"predictions": "/data/preds.csv", "output_dir": "/data/report"}

// ml_reduce_dimensions
{"data": "/data/features.csv", "output": "/data/pca.png",
 "method": "pca", "label_col": "target"}
```

The `predictions` table needs `y_true` and `y_pred` columns; probabilities are
optional via a `y_proba` column (binary) or `y_proba_0`, `y_proba_1`, …
(multiclass).

## Parity & scope

Each MCP tool maps 1:1 to a `scitex-ml` CLI verb with the same name and JSON
shape (`compute-metrics` ↔ `ml_compute_metrics`, …). The package exposes only
this **stateless** analysis slice via MCP — training loops, optimizers,
EarlyStopping and the deep submodule API are Python-only, so scitex-ml declares
`mcp_parity_exempt = true` under `[tool.scitex_dev]`.

## See also

- [14_cli-reference.md](04_cli-reference.md) — the identical CLI surface
- `general/03_interface/03_mcp/` — ecosystem MCP conventions
