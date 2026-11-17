#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-09-21 20:20:00 (ywatanabe)"
# File: timeseries_cv_demo.py

"""
Examples demonstrating the time series cross-validation modules.

This script shows how to use:
1. Individual time series CV splitters
2. The intelligent TimeSeriesCVCoordinator
3. Integration with classification reporters
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import scitex_logging as slogging

_console = slogging.getConsole(__name__, level=slogging.INFO)


def generate_synthetic_timeseries(
    n_samples: int = 1000,
    n_features: int = 10,
    n_groups: int = None,
    noise_level: float = 0.1,
    imbalance_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data for demonstration.

    Returns
    -------
    X : Features
    y : Labels (binary)
    timestamps : Time points
    groups : Group labels (if n_groups specified)
    """
    np.random.seed(42)

    # Generate features with temporal correlation
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        # AR(1) process with different parameters
        phi = 0.3 + i * 0.05  # Autocorrelation
        X[:, i] = np.random.randn(n_samples)
        for t in range(1, n_samples):
            X[t, i] = phi * X[t - 1, i] + np.sqrt(1 - phi**2) * np.random.randn()

    # Add noise
    X += np.random.randn(n_samples, n_features) * noise_level

    # Generate labels with temporal pattern
    # Early timepoints more likely to be class 0, later more likely class 1
    transition_point = n_samples // 2
    probabilities = np.zeros(n_samples)
    probabilities[:transition_point] = imbalance_ratio
    probabilities[transition_point:] = 1 - imbalance_ratio

    # Add some temporal structure
    for t in range(1, n_samples):
        probabilities[t] = 0.7 * probabilities[t] + 0.3 * probabilities[t - 1]

    y = (np.random.rand(n_samples) < probabilities).astype(int)

    # Generate timestamps
    timestamps = np.arange(n_samples) + np.random.randn(n_samples) * 0.01
    timestamps = np.sort(timestamps)

    # Generate groups if requested
    if n_groups:
        groups = np.repeat(np.arange(n_groups), n_samples // n_groups)
        if len(groups) < n_samples:
            groups = np.concatenate(
                [groups, [n_groups - 1] * (n_samples - len(groups))]
            )
    else:
        groups = None

    return X, y, timestamps, groups


def demo_basic_splitters():
    """Demonstrate basic time series CV splitters."""
    _console.info("=" * 70)
    _console.info("DEMO: Basic Time Series CV Splitters")
    _console.info("=" * 70)

    from scitex_ml.classification import (
        BlockingTimeSeriesSplit,
        SlidingWindowSplit,
        StratifiedTimeSeriesSplit,
    )

    # Generate data
    X, y, timestamps, _ = generate_synthetic_timeseries(n_samples=500)

    _console.info("\n1. StratifiedTimeSeriesSplit (maintains class balance)")
    _console.info("-" * 40)
    tscv = StratifiedTimeSeriesSplit(n_splits=3, test_ratio=0.2)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X, y, timestamps)):
        train_classes, train_counts = np.unique(y[train_idx], return_counts=True)
        test_classes, test_counts = np.unique(y[test_idx], return_counts=True)

        _console.info(f"Fold {fold}:")
        _console.info(
            f"  Train: {len(train_idx)} samples - Class dist: {dict(zip(train_classes, train_counts))}"
        )
        _console.info(
            f"  Test:  {len(test_idx)} samples - Class dist: {dict(zip(test_classes, test_counts))}"
        )
        _console.info(
            f"  Time range - Train: [{timestamps[train_idx].min():.1f}, {timestamps[train_idx].max():.1f}]"
        )
        _console.info(
            f"  Time range - Test:  [{timestamps[test_idx].min():.1f}, {timestamps[test_idx].max():.1f}]"
        )

    _console.info("\n2. SlidingWindowSplit (fixed-size windows)")
    _console.info("-" * 40)
    swcv = SlidingWindowSplit(window_size=200, step_size=50, test_size=50)

    for fold, (train_idx, test_idx) in enumerate(swcv.split(X, y, timestamps)):
        if fold >= 3:  # Only show first 3 folds
            _console.info(f"... ({swcv.get_n_splits(X) - 3} more folds)")
            break
        _console.info(f"Fold {fold}:")
        _console.info(f"  Train: {len(train_idx)} samples")
        _console.info(f"  Test:  {len(test_idx)} samples")

    # Generate data with groups
    _console.info("\n3. BlockingTimeSeriesSplit (for multiple time series)")
    _console.info("-" * 40)
    X_grouped, y_grouped, timestamps_grouped, groups = generate_synthetic_timeseries(
        n_samples=500, n_groups=5
    )

    btscv = BlockingTimeSeriesSplit(n_splits=3)

    for fold, (train_idx, test_idx) in enumerate(
        btscv.split(X_grouped, y_grouped, timestamps_grouped, groups)
    ):
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])

        _console.info(f"Fold {fold}:")
        _console.info(f"  Train: {len(train_idx)} samples from groups {train_groups}")
        _console.info(f"  Test:  {len(test_idx)} samples from groups {test_groups}")


def demo_coordinator():
    """Demonstrate the intelligent TimeSeriesCVCoordinator."""
    _console.info("\n" + "=" * 70)
    _console.info("DEMO: TimeSeriesCVCoordinator (Automatic Strategy Selection)")
    _console.info("=" * 70)

    from scitex_ml.classification import TimeSeriesCVCoordinator

    scenarios = [
        ("Small dataset", 50, 5, None),
        ("Large dataset", 10000, 20, None),
        ("Imbalanced classes", 1000, 10, None),
        ("Multiple patients", 1000, 10, 10),
    ]

    for scenario_name, n_samples, n_features, n_groups in scenarios:
        _console.info(f"\n{scenario_name}")
        _console.info("-" * 40)

        # Generate appropriate data
        if scenario_name == "Imbalanced classes":
            X, y, timestamps, groups = generate_synthetic_timeseries(
                n_samples, n_features, n_groups, imbalance_ratio=0.2
            )
        else:
            X, y, timestamps, groups = generate_synthetic_timeseries(
                n_samples, n_features, n_groups
            )

        # Create coordinator with auto strategy
        coordinator = TimeSeriesCVCoordinator(strategy="auto", n_splits=3, verbose=True)

        # Analyze data
        metadata = coordinator.analyze_data(X, y, timestamps, groups)

        _console.info(f"\nData characteristics:")
        _console.info(f"  Samples: {metadata.n_samples}")
        _console.info(f"  Features: {metadata.n_features}")
        _console.info(f"  Classes: {metadata.n_classes}")
        _console.info(f"  Has groups: {metadata.has_groups}")
        _console.info(f"  Is balanced: {metadata.is_balanced}")

        # Generate splits
        _console.info(f"\nCV splits:")
        for fold, (train_idx, test_idx) in enumerate(
            coordinator.split(X, y, timestamps, groups)
        ):
            _console.info(f"  Fold {fold}: Train={len(train_idx)}, Test={len(test_idx)}")

            # Only show first fold for large datasets
            if n_samples > 5000:
                _console.info("  (skipping remaining folds for large dataset)")
                break


def demo_with_classifier():
    """Demonstrate integration with actual classification."""
    _console.info("\n" + "=" * 70)
    _console.info("DEMO: Integration with Classification")
    _console.info("=" * 70)

    from scitex_ml.classification import (
        SingleTaskClassificationReporter,
        TimeSeriesCVCoordinator,
    )

    # Generate data
    X, y, timestamps, _ = generate_synthetic_timeseries(n_samples=1000)

    # Setup coordinator
    coordinator = TimeSeriesCVCoordinator(
        strategy="stratified",
        n_splits=5,
        test_ratio=0.2,
        gap="10s",  # 10 second gap (assuming 1 sample per second)
        verbose=False,
    )

    # Initialize reporter
    reporter = SingleTaskClassificationReporter(
        name="timeseries_demo", output_dir="./ts_cv_demo_results"
    )

    # Train and evaluate
    scores = []

    _console.info("\nRunning cross-validation with LogisticRegression...")
    for fold, (train_idx, test_idx) in enumerate(coordinator.split(X, y, timestamps)):
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X[train_idx], y[train_idx])

        # Predict
        y_pred = model.predict(X[test_idx])
        y_proba = model.predict_proba(X[test_idx])

        # Calculate metrics
        ba = balanced_accuracy_score(y[test_idx], y_pred)
        scores.append(ba)

        _console.info(f"  Fold {fold}: BA = {ba:.3f}")

        # Save metrics (if reporter is available)
        try:
            reporter.calc_bacc(y[test_idx], y_pred, fold)
            reporter.calc_conf_mat(y[test_idx], y_pred, fold)
            reporter.calc_roc_auc(y[test_idx], y_proba[:, 1], fold)
        except Exception as e:
            pass  # Reporter may not be fully configured

    _console.info(f"\nOverall performance: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


def demo_train_val_test_split():
    """Demonstrate train/validation/test splitting."""
    _console.info("\n" + "=" * 70)
    _console.info("DEMO: Train/Validation/Test Splitting")
    _console.info("=" * 70)

    from scitex_ml.classification import TimeSeriesCVCoordinator

    # Generate data
    X, y, timestamps, _ = generate_synthetic_timeseries(n_samples=1000)

    # Setup coordinator with validation split
    coordinator = TimeSeriesCVCoordinator(
        strategy="stratified", n_splits=3, test_ratio=0.2, val_ratio=0.1, verbose=False
    )

    _console.info("\nGenerating train/val/test splits...")
    _console.info("-" * 40)

    for fold, (train_idx, val_idx, test_idx) in enumerate(
        coordinator.split_with_validation(X, y, timestamps)
    ):
        _console.info(f"Fold {fold}:")
        _console.info(
            f"  Train: {len(train_idx)} samples ({len(train_idx) / len(X) * 100:.1f}%)"
        )
        _console.info(f"  Val:   {len(val_idx)} samples ({len(val_idx) / len(X) * 100:.1f}%)")
        _console.info(f"  Test:  {len(test_idx)} samples ({len(test_idx) / len(X) * 100:.1f}%)")

        # Check temporal ordering
        train_max_time = timestamps[train_idx].max()
        val_min_time = timestamps[val_idx].min() if len(val_idx) > 0 else float("inf")
        val_max_time = timestamps[val_idx].max() if len(val_idx) > 0 else -float("inf")
        test_min_time = (
            timestamps[test_idx].min() if len(test_idx) > 0 else float("inf")
        )

        _console.info(
            f"  Temporal check: Train ends at {train_max_time:.1f}, "
            f"Val starts at {val_min_time:.1f}, Test starts at {test_min_time:.1f}"
        )

        # Verify no overlap
        assert train_max_time <= val_min_time, "Train/Val overlap detected!"
        assert val_max_time <= test_min_time, "Val/Test overlap detected!"
        _console.info("  ✓ No temporal overlap detected")


def visualize_cv_splits():
    """Visualize the different CV strategies."""
    _console.info("\n" + "=" * 70)
    _console.info("VISUALIZATION: CV Split Strategies")
    _console.info("=" * 70)

    from scitex_ml.classification import (
        SlidingWindowSplit,
        StratifiedTimeSeriesSplit,
        TimeSeriesCVCoordinator,
    )

    # Generate data
    n_samples = 200
    X, y, timestamps, _ = generate_synthetic_timeseries(n_samples=n_samples)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    strategies = [
        ("Stratified Time Series", StratifiedTimeSeriesSplit(n_splits=3)),
        (
            "Sliding Window",
            SlidingWindowSplit(window_size=80, step_size=40, test_size=40),
        ),
        (
            "Expanding Window (via Coordinator)",
            TimeSeriesCVCoordinator(strategy="expanding", n_splits=3),
        ),
    ]

    for ax_idx, (name, splitter) in enumerate(strategies):
        ax = axes[ax_idx]

        # Generate splits
        if isinstance(splitter, TimeSeriesCVCoordinator):
            splits = list(splitter.split(X, y, timestamps))
        else:
            splits = list(splitter.split(X, y, timestamps))

        # Visualize
        for fold, (train_idx, test_idx) in enumerate(splits[:5]):  # Max 5 folds
            # Create a timeline
            timeline = np.zeros(n_samples)
            timeline[train_idx] = 1  # Train
            timeline[test_idx] = 2  # Test

            # Plot as horizontal bars
            y_pos = fold
            for i in range(n_samples):
                if timeline[i] == 1:
                    ax.barh(y_pos, 1, left=i, height=0.8, color="blue", alpha=0.6)
                elif timeline[i] == 2:
                    ax.barh(y_pos, 1, left=i, height=0.8, color="red", alpha=0.6)

        ax.set_xlim(0, n_samples)
        ax.set_ylim(-0.5, min(5, len(splits)) - 0.5)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Fold")
        ax.set_title(name)
        ax.set_yticks(range(min(5, len(splits))))

        # Add legend
        if ax_idx == 0:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="blue", alpha=0.6, label="Train"),
                Patch(facecolor="red", alpha=0.6, label="Test"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save figure
    output_path = "./ts_cv_splits_visualization.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    _console.info(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    """Run all demonstrations."""

    # Basic demonstrations
    demo_basic_splitters()
    demo_coordinator()
    demo_with_classifier()
    demo_train_val_test_split()

    # Visualization
    try:
        visualize_cv_splits()
    except Exception as e:
        _console.info(f"\nVisualization skipped: {e}")

    _console.info("\n" + "=" * 70)
    _console.info("All demonstrations completed successfully!")
    _console.info("=" * 70)


if __name__ == "__main__":
    main()
