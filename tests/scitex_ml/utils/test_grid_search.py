#!/usr/bin/env python3
"""
Comprehensive test suite for grid search functionality.

This test module verifies:
- Grid parameter generation
- Random shuffling of combinations
- Counting grid combinations
- Performance with large parameter spaces
- Integration with ML workflows
- Memory efficiency
"""

import random

import numpy as np
import pytest

from scitex_ml.utils.grid_search import count_grids, yield_grids


class TestGridSearch:
    """Test cases for grid search functionality."""

    @pytest.fixture
    def simple_grid(self):
        """Simple parameter grid for testing."""
        return {"param1": [1, 2, 3], "param2": ["a", "b"], "param3": [True, False]}

    @pytest.fixture
    def ml_grid(self):
        """Machine learning parameter grid."""
        return {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64],
            "dropout": [0.0, 0.2, 0.5],
            "optimizer": ["adam", "sgd"],
        }

    @pytest.fixture
    def large_grid(self):
        """Large parameter grid for performance testing."""
        return {
            "param1": list(range(10)),
            "param2": list(range(10)),
            "param3": list(range(10)),
            "param4": ["a", "b", "c"],
        }

    def test_yield_grids_basic_returns_expected_count(self, simple_grid):
        """Test that yield_grids returns the cartesian-product count."""
        # Arrange
        expected = 3 * 2 * 2

        # Act
        combinations = list(yield_grids(simple_grid))

        # Assert
        assert len(combinations) == expected

    def test_yield_grids_basic_each_combination_is_dict(self, simple_grid):
        """Test that each yielded combination is a dict."""
        # Arrange
        combinations = list(yield_grids(simple_grid))

        # Act
        types_seen = {type(c) for c in combinations}

        # Assert
        assert types_seen == {dict}

    def test_yield_grids_basic_each_combination_has_all_keys(self, simple_grid):
        """Test that each yielded combination has all grid keys."""
        # Arrange
        expected_keys = set(simple_grid.keys())

        # Act
        combinations = list(yield_grids(simple_grid))

        # Assert
        assert all(set(c.keys()) == expected_keys for c in combinations)

    def test_yield_grids_values_match_manual_cartesian_product(self, simple_grid):
        """Test that yielded combinations match a manual cartesian-product reference."""
        # Arrange
        expected = []
        for p1 in [1, 2, 3]:
            for p2 in ["a", "b"]:
                for p3 in [True, False]:
                    expected.append({"param1": p1, "param2": p2, "param3": p3})

        def sort_key(d):
            return (d["param1"], d["param2"], d["param3"])

        expected.sort(key=sort_key)

        # Act
        combinations = sorted(yield_grids(simple_grid), key=sort_key)

        # Assert
        assert combinations == expected

    def test_yield_grids_random_produces_same_set_as_ordered(self, simple_grid):
        """Test that random=True yields the same set of combinations as ordered."""
        # Arrange
        random.seed(42)
        random_combos = list(yield_grids(simple_grid, random=True))
        ordered_combos = list(yield_grids(simple_grid, random=False))

        # Act
        random_set = {tuple(sorted(d.items())) for d in random_combos}
        ordered_set = {tuple(sorted(d.items())) for d in ordered_combos}

        # Assert
        assert random_set == ordered_set

    def test_count_grids_returns_cartesian_product_size(self, simple_grid):
        """Test that count_grids returns the product of value-list lengths."""
        # Arrange
        expected = 12

        # Act
        count = count_grids(simple_grid)

        # Assert
        assert count == expected

    def test_count_grids_matches_actual_generation(self, simple_grid):
        """Test that count_grids matches len(list(yield_grids))."""
        # Arrange
        combinations = list(yield_grids(simple_grid))

        # Act
        count = count_grids(simple_grid)

        # Assert
        assert count == len(combinations)

    def test_count_grids_empty_grid_returns_one(self):
        """Test that count_grids on empty grid returns 1 (empty product)."""
        # Arrange
        empty_grid = {}

        # Act
        result = count_grids(empty_grid)

        # Assert
        assert result == 1

    def test_count_grids_single_param_returns_param_length(self):
        """Test that single-parameter grid returns the parameter's length."""
        # Arrange
        grid = {"p1": [1, 2, 3]}

        # Act
        result = count_grids(grid)

        # Assert
        assert result == 3

    def test_count_grids_multi_param_returns_product(self):
        """Test that count_grids returns the product across multiple params."""
        # Arrange
        grid = {"p1": [1, 2], "p2": [1, 2, 3], "p3": [1, 2, 3, 4]}

        # Act
        result = count_grids(grid)

        # Assert
        assert result == 2 * 3 * 4

    def test_empty_grid_yields_one_empty_dict(self):
        """Test that an empty grid yields exactly one empty dict."""
        # Arrange
        empty_grid = {}

        # Act
        combinations = list(yield_grids(empty_grid))

        # Assert
        assert combinations == [{}]

    def test_single_value_parameters_yields_one_combination(self):
        """Test that grid with single-value params yields exactly one combination."""
        # Arrange
        grid = {"param1": [1], "param2": ["only"], "param3": [True]}

        # Act
        combinations = list(yield_grids(grid))

        # Assert
        assert combinations == [{"param1": 1, "param2": "only", "param3": True}]

    def test_ml_parameter_grid_returns_expected_count(self, ml_grid):
        """Test that ML grid yields 3*3*3*2 = 54 combinations."""
        # Arrange
        expected = 3 * 3 * 3 * 2

        # Act
        count = count_grids(ml_grid)

        # Assert
        assert count == expected

    def test_ml_parameter_grid_first_combo_uses_first_values(self, ml_grid):
        """Test that the first yielded combination uses the first value of each list."""
        # Arrange
        expected_first = {
            "learning_rate": 0.001,
            "batch_size": 16,
            "dropout": 0.0,
            "optimizer": "adam",
        }

        # Act
        first = next(iter(yield_grids(ml_grid)))

        # Assert
        assert first == expected_first

    def test_mixed_types_grid_returns_expected_count(self):
        """Test that a mixed-type grid yields the product of value-list lengths."""
        # Arrange
        grid = {
            "int_param": [1, 2, 3],
            "float_param": [0.1, 0.2],
            "str_param": ["a", "b"],
            "bool_param": [True, False],
            "none_param": [None, "value"],
            "list_param": [[1, 2], [3, 4]],
        }

        # Act
        combinations = list(yield_grids(grid))

        # Assert
        assert len(combinations) == 3 * 2 * 2 * 2 * 2 * 2

    def test_mixed_types_grid_preserves_value_types(self):
        """Test that mixed-type grid combinations contain values from each list."""
        # Arrange
        grid = {
            "int_param": [1, 2, 3],
            "float_param": [0.1, 0.2],
            "str_param": ["a", "b"],
            "bool_param": [True, False],
            "none_param": [None, "value"],
            "list_param": [[1, 2], [3, 4]],
        }

        # Act
        combinations = list(yield_grids(grid))

        # Assert
        assert all(
            (
                combo["int_param"] in [1, 2, 3]
                and combo["float_param"] in [0.1, 0.2]
                and combo["str_param"] in ["a", "b"]
                and combo["bool_param"] in [True, False]
                and combo["none_param"] in [None, "value"]
                and combo["list_param"] in [[1, 2], [3, 4]]
            )
            for combo in combinations
        )

    def test_generator_efficiency_is_iterable(self, large_grid):
        """Test that yield_grids returns an iterable object."""
        # Arrange
        gen = yield_grids(large_grid)

        # Act
        is_iterable = hasattr(gen, "__iter__") and hasattr(gen, "__next__")

        # Assert
        assert is_iterable

    def test_generator_efficiency_supports_partial_consumption(self, large_grid):
        """Test that we can consume only the first few items from the generator."""
        # Arrange
        gen = yield_grids(large_grid)

        # Act
        first_five = []
        for i, combo in enumerate(gen):
            first_five.append(combo)
            if i >= 4:
                break

        # Assert
        assert len(first_five) == 5

    def test_large_grid_performance_count_is_correct(self):
        """Test that count_grids on a large grid returns the expected product."""
        # Arrange
        grid = {f"param_{i}": list(range(5)) for i in range(8)}

        # Act
        count = count_grids(grid)

        # Assert
        assert count == 5**8

    def test_deterministic_order_is_reproducible(self, simple_grid):
        """Test that non-random generation produces the same order across calls."""
        # Arrange
        combos1 = list(yield_grids(simple_grid, random=False))

        # Act
        combos2 = list(yield_grids(simple_grid, random=False))

        # Assert
        assert combos1 == combos2

    def test_random_different_seeds_produce_same_combination_set(self, simple_grid):
        """Test that different seeds yield the same set of combinations."""
        # Arrange
        random.seed(42)
        combos1 = list(yield_grids(simple_grid, random=True))
        random.seed(123)
        combos2 = list(yield_grids(simple_grid, random=True))

        # Act
        set1 = {tuple(sorted(d.items())) for d in combos1}
        set2 = {tuple(sorted(d.items())) for d in combos2}

        # Assert
        assert set1 == set2

    def test_nested_parameter_values_yields_expected_count(self):
        """Test that nested-dict parameter values yield expected count."""
        # Arrange
        grid = {
            "model_config": [{"layers": 2, "units": 64}, {"layers": 3, "units": 128}],
            "training_config": [
                {"epochs": 10, "patience": 3},
                {"epochs": 20, "patience": 5},
            ],
        }

        # Act
        combinations = list(yield_grids(grid))

        # Assert
        assert len(combinations) == 4

    def test_nested_parameter_values_preserve_inner_keys(self):
        """Test that nested dict values keep their inner keys intact."""
        # Arrange
        grid = {
            "model_config": [{"layers": 2, "units": 64}, {"layers": 3, "units": 128}],
            "training_config": [
                {"epochs": 10, "patience": 3},
                {"epochs": 20, "patience": 5},
            ],
        }

        # Act
        combinations = list(yield_grids(grid))

        # Assert
        assert all(
            (
                "layers" in combo["model_config"]
                and "units" in combo["model_config"]
                and "epochs" in combo["training_config"]
                and "patience" in combo["training_config"]
            )
            for combo in combinations
        )

    def test_sklearn_compatibility_yields_same_combinations(self, ml_grid):
        """Test that yield_grids produces the same combinations as sklearn ParameterGrid."""
        # Arrange
        from sklearn.model_selection import ParameterGrid

        our_set = {tuple(sorted(d.items())) for d in yield_grids(ml_grid)}
        sklearn_set = {tuple(sorted(d.items())) for d in ParameterGrid(ml_grid)}

        # Act
        equal_sets = our_set == sklearn_set

        # Assert
        assert equal_sets

    def test_memory_efficiency_returns_value_references(self):
        """Test that generator returns references to value-list elements, not copies."""
        # Arrange
        grid = {
            "data": [np.zeros((100, 100)), np.ones((100, 100))],
            "scale": [0.1, 1.0],
        }
        gen = yield_grids(grid)

        # Act
        first = next(gen)

        # Assert
        assert first["data"] is grid["data"][0]

    def test_parameter_filtering_drops_invalid_combos(self, ml_grid):
        """Test that filtering combinations preserves only valid ones."""
        # Arrange
        valid_combos = []
        for combo in yield_grids(ml_grid):
            if combo["optimizer"] == "sgd" and combo["learning_rate"] > 0.01:
                continue
            valid_combos.append(combo)

        # Act
        all_valid = all(
            (combo["optimizer"] != "sgd") or (combo["learning_rate"] <= 0.01)
            for combo in valid_combos
        )

        # Assert
        assert all_valid


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/grid_search.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-22 23:54:02"
# # Author: Yusuke Watanabe (ywatanabe@scitex_ml)
#
# """
# This script defines scitex_ml.utils.grid_search
# """
#
# # Imports
# import itertools as _itertools
# import random as _random
# import sys as _sys
#
# import matplotlib.pyplot as _plt
# import scitex as _scitex
#
#
# # Functions
# def yield_grids(params_grid: dict, random=False):
#     combinations = list(_itertools.product(*params_grid.values()))
#     if random:
#         _random.shuffle(combinations)
#     for values in combinations:
#         yield dict(zip(params_grid.keys(), values))
#
#
# def count_grids(params_grid):
#     num_combinations = 1
#     for values in params_grid.values():
#         num_combinations *= len(values)
#     return num_combinations
#
# # --------------------------------------------------------------------------------
# # End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/grid_search.py
# # --------------------------------------------------------------------------------
