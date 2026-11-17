#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2024-11-20 08:49:50 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/ml/training/_LearningCurveLogger.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionality:
    - Records and visualizes learning curves during model training
    - Supports tracking of multiple metrics across training/validation/test phases
    - Generates plots showing training progress over iterations and epochs
    - Delegates plotting to scitex_ml.plt.plot_learning_curve for consistency

Input:
    - Training metrics dictionary containing loss, accuracy, predictions etc.
    - Step information (Training/Validation/Test)

Output:
    - Learning curve plots via scitex_ml.plt.plot_learning_curve
    - DataFrames with recorded metrics
    - Training progress prints

Prerequisites:
    - PyTorch
    - scikit-learn
    - matplotlib
    - pandas
    - numpy
    - scitex
"""

import re
import warnings
from collections import defaultdict
from pprint import pformat
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scitex_logging as slogging

_console = slogging.getConsole(__name__, level=slogging.INFO)


class LearningCurveLogger:
    """Records and visualizes learning metrics during model training.

    Example
    -------
    >>> logger = LearningCurveLogger()
    >>> metrics = {
    ...     "loss_plot": 0.5,
    ...     "balanced_ACC_plot": 0.8,
    ...     "pred_proba": pred_proba,
    ...     "true_class": labels,
    ...     "i_fold": 0,
    ...     "i_epoch": 1,
    ...     "i_global": 100
    ... }
    >>> logger(metrics, "Training")
    >>> fig = logger.plot_learning_curves()
    """

    def __init__(self) -> None:
        self.logged_dict: Dict[str, Dict] = defaultdict(dict)

        warnings.warn(
            '\n"gt_label" will be removed in the future. Please use "true_class" instead.\n',
            DeprecationWarning,
        )

    def __call__(self, dict_to_log: Dict[str, Any], step: str) -> None:
        """Logs metrics for a training step.

        Parameters
        ----------
        dict_to_log : Dict[str, Any]
            Dictionary containing metrics to log
        step : str
            Phase of training ('Training', 'Validation', or 'Test')
        """
        # Handle deprecated gt_label
        if "gt_label" in dict_to_log:
            dict_to_log["true_class"] = dict_to_log.pop("gt_label")

        for k_to_log in dict_to_log:
            try:
                self.logged_dict[step][k_to_log].append(dict_to_log[k_to_log])
            except KeyError:
                self.logged_dict[step][k_to_log] = [dict_to_log[k_to_log]]

    @property
    def dfs(self) -> Dict[str, pd.DataFrame]:
        """Returns DataFrames of logged metrics.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of DataFrames for each step
        """
        return self._to_dfs_pivot(self.logged_dict, pivot_column=None)

    def to_metrics_df(self) -> pd.DataFrame:
        """Convert logged data to metrics DataFrame for plot_learning_curve.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: step, i_global, i_epoch, i_batch, and metric columns
        """
        all_rows = []

        for step, metrics in self.logged_dict.items():
            n_samples = len(metrics["i_global"])

            for i in range(n_samples):
                row = {"step": step}
                for key, values in metrics.items():
                    # Only include scalar metrics for the DataFrame
                    if key.endswith("_plot") or key in [
                        "i_global",
                        "i_epoch",
                        "i_batch",
                        "i_fold",
                    ]:
                        row[key] = values[i]
                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        # Rename _plot columns
        df.columns = [
            col.replace("_plot", "") if col.endswith("_plot") else col
            for col in df.columns
        ]

        # Ensure i_batch exists (create dummy if not logged)
        if "i_batch" not in df.columns:
            df["i_batch"] = df.groupby(["step", "i_epoch"]).cumcount()

        return df

    def plot_learning_curves(
        self,
        title: Optional[str] = None,
        max_n_ticks: int = 4,
        linewidth: float = 1,
        scattersize: float = 3,
        yscale: str = "linear",
        spath: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Plots learning curves from logged metrics.

        Delegates to scitex_ml.plt.plot_learning_curve for consistent plotting.

        Parameters
        ----------
        title : str, optional
            Plot title
        max_n_ticks : int
            Maximum number of ticks on axes
        linewidth : float
            Width of plot lines
        scattersize : float
            Size of scatter points
        yscale : str
            Y-axis scale ('linear' or 'log')
        spath : str, optional
            Save path for the figure

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing learning curves
        """
        from scitex_ml.plt import plot_learning_curve

        # Convert to metrics DataFrame
        metrics_df = self.to_metrics_df()

        # Find keys to plot (exclude metadata columns)
        keys_to_plot = [
            col
            for col in metrics_df.columns
            if col not in ["step", "i_global", "i_epoch", "i_batch", "i_fold"]
        ]

        if len(keys_to_plot) == 0:
            # Create empty plot when no plot keys found
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("No metrics to plot")
            if title:
                fig.text(0.5, 0.95, title, ha="center")
            return fig

        # Delegate to centralized plotting function
        fig = plot_learning_curve(
            metrics_df=metrics_df,
            keys=keys_to_plot,
            title=title or "Learning Curves",
            max_n_ticks=max_n_ticks,
            scattersize=scattersize,
            linewidth=linewidth,
            yscale=yscale,
            spath=spath,
        )

        return fig

    def get_x_of_i_epoch(self, x: str, step: str, i_epoch: int) -> np.ndarray:
        """Gets metric values for a specific epoch.

        Parameters
        ----------
        x : str
            Name of metric to retrieve
        step : str
            Training phase
        i_epoch : int
            Epoch number

        Returns
        -------
        np.ndarray
            Array of metric values for specified epoch
        """
        indi = np.array(self.logged_dict[step]["i_epoch"]) == i_epoch
        x_all_arr = np.array(self.logged_dict[step][x])
        assert len(indi) == len(x_all_arr)
        return x_all_arr[indi]

    def print(self, step: str) -> None:
        """Prints metrics for given step.

        Parameters
        ----------
        step : str
            Training phase to print metrics for
        """
        df_pivot_i_epoch = self._to_dfs_pivot(self.logged_dict, pivot_column="i_epoch")
        df_pivot_i_epoch_step = df_pivot_i_epoch[step]
        df_pivot_i_epoch_step.columns = self._rename_if_key_to_plot(
            df_pivot_i_epoch_step.columns
        )
        _console.info("\n----------------------------------------\n")
        _console.info(f"\n{step}: (mean of batches)\n")
        _console.info(pformat(df_pivot_i_epoch_step))
        _console.info("\n----------------------------------------\n")

    @staticmethod
    def _rename_if_key_to_plot(x: Union[str, pd.Index]) -> Union[str, pd.Index]:
        """Rename metric keys for plotting.

        Parameters
        ----------
        x : str or pd.Index
            Metric name(s) to rename

        Returns
        -------
        str or pd.Index
            Renamed metric name(s)
        """
        if isinstance(x, str):
            if re.search("_plot$", x):
                return x.replace("_plot", "")
            else:
                return x
        else:
            return x.str.replace("_plot", "")

    @staticmethod
    def _to_dfs_pivot(
        logged_dict: Dict[str, Dict],
        pivot_column: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Convert logged dictionary to pivot DataFrames.

        Parameters
        ----------
        logged_dict : Dict[str, Dict]
            Dictionary of logged metrics
        pivot_column : str, optional
            Column to pivot on

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of pivot DataFrames
        """
        dfs_pivot = {}
        for step_k in logged_dict.keys():
            if pivot_column is None:
                df = pd.DataFrame(logged_dict[step_k])
            else:
                df = (
                    pd.DataFrame(logged_dict[step_k])
                    .groupby(pivot_column)
                    .mean()
                    .reset_index()
                    .set_index(pivot_column)
                )
            dfs_pivot[step_k] = df
        return dfs_pivot


# EOF
