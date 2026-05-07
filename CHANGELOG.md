# Changelog

All notable changes to `scitex-ml` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
