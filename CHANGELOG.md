# Changelog

All notable changes to `scitex-ml` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-05-27

### Added

- **Stateless analysis CLI + MCP server.** New `scitex-ml` console script and `scitex_ml._mcp_server` FastMCP server exposing three file-in → JSON/artifact-out verbs with identical CLI↔MCP JSON: `compute-metrics` ↔ `ml_compute_metrics`, `generate-report` ↔ `ml_generate_report`, `reduce-dimensions` ↔ `ml_reduce_dimensions`. Plus `list-python-apis`, the `mcp` and `skills` command groups, and shell completion. Adds Sphinx `cli`/`mcp` pages and the `04_cli-reference` / `05_mcp-tools` skill leaves. MCP parity is scoped via `audit.mcp-tools-allowlist` (scitex-dev ≥ 0.13.0).

### Changed

- `import scitex_ml` is now lazy (PEP 562 `__getattr__`): cold-start drops from ~7.9s to ~76ms so CLI tab-completion stays fast. `[heavy]` deps remain gracefully optional via `try_import_optional`.

### Added

- Initial factor-out from `scitex.ai` (umbrella scitex-python package) into a
  standalone `scitex-ml` package. Mirrors the public API of the in-umbrella
  module 1:1; the umbrella's `scitex.ai` becomes a thin re-export bridge.
- Submodules: `activation`, `classification`, `clustering`,
  `feature_extraction`, `feature_selection`, `_gen_ai` (lazy `GenAI`),
  `loss`, `metrics`, `optim`, `plt`, `sampling`, `sk`, `sklearn`,
  `training`, `utils`.
- Public API: `ClassificationReporter`, `Classifier`, `EarlyStopping`,
  `LearningCurveLogger`, `MultiTaskLoss`, `GenAI` (lazy),
  `get_optimizer`, `set_optimizer`.
