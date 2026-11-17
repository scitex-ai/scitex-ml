#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo: train a small MNIST perceptron and plot its LearningCurveLogger curves.

Run with ``python -m scitex_ml.training._learning_curve_logger_demo``.
Needs the optional torch / torchvision stack: ``pip install "scitex-ml[all]"``.
"""

from __future__ import annotations

import argparse
import os
import warnings

from scitex_ml._optional import missing_optional_dependency
from scitex_ml.training._LearningCurveLogger import LearningCurveLogger

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)



def main(args):
    """Demo learning curve logger with MNIST training."""
    from sklearn.metrics import balanced_accuracy_score

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from torch.utils.data.dataset import Subset
        from torchvision import datasets
    except ImportError as exc:
        raise missing_optional_dependency(
            "scitex_ml.training LearningCurveLogger demo", exc
        ) from exc

    ################################################################################
    ## NN
    ################################################################################
    class Perceptron(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(28 * 28, 50)
            self.l2 = nn.Linear(50, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.l1(x)
            x = self.l2(x)
            return x

    ################################################################################
    ## Prepare demo data
    ################################################################################
    ## Downloads
    _ds_tra_val = datasets.MNIST("/tmp/mnist", train=True, download=True)
    _ds_tes = datasets.MNIST("/tmp/mnist", train=False, download=True)

    ## Training-Validation splitting
    n_samples = len(_ds_tra_val)
    train_size = int(n_samples * 0.8)

    subset1_indices = list(range(0, train_size))
    subset2_indices = list(range(train_size, n_samples))

    _ds_tra = Subset(_ds_tra_val, subset1_indices)
    _ds_val = Subset(_ds_tra_val, subset2_indices)

    ## to tensors
    ds_tra = TensorDataset(
        _ds_tra.dataset.data.to(torch.float32),
        _ds_tra.dataset.targets,
    )
    ds_val = TensorDataset(
        _ds_val.dataset.data.to(torch.float32),
        _ds_val.dataset.targets,
    )
    ds_tes = TensorDataset(
        _ds_tes.data.to(torch.float32),
        _ds_tes.targets,
    )

    ## to dataloaders
    batch_size = 64
    dl_tra = DataLoader(
        dataset=ds_tra,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    dl_tes = DataLoader(
        dataset=ds_tes,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    ################################################################################
    ## Preparation
    ################################################################################
    model = Perceptron()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    softmax = nn.Softmax(dim=-1)

    ################################################################################
    ## Main
    ################################################################################
    lc_logger = LearningCurveLogger()
    i_global = 0

    i_fold = 0
    max_epochs = 3

    for i_epoch in range(max_epochs):
        step = "Validation"
        for i_batch, batch in enumerate(dl_val):
            X, T = batch
            logits = model(X)
            pred_proba = softmax(logits)
            pred_class = pred_proba.argmax(dim=-1)
            loss = loss_func(logits, T)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bACC = balanced_accuracy_score(T, pred_class)

            dict_to_log = {
                "loss_plot": float(loss),
                "balanced_ACC_plot": float(bACC),
                "pred_proba": pred_proba.detach().cpu().numpy(),
                "true_class": T.cpu().numpy(),
                "i_fold": i_fold,
                "i_epoch": i_epoch,
                "i_batch": i_batch,
                "i_global": i_global,
            }
            lc_logger(dict_to_log, step)

        lc_logger.print(step)

        step = "Training"
        for i_batch, batch in enumerate(dl_tra):
            optimizer.zero_grad()

            X, T = batch
            logits = model(X)
            pred_proba = softmax(logits)
            pred_class = pred_proba.argmax(dim=-1)
            loss = loss_func(logits, T)

            loss.backward()
            optimizer.step()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bACC = balanced_accuracy_score(T, pred_class)

            dict_to_log = {
                "loss_plot": float(loss),
                "balanced_ACC_plot": float(bACC),
                "pred_proba": pred_proba.detach().cpu().numpy(),
                "true_class": T.cpu().numpy(),
                "i_fold": i_fold,
                "i_epoch": i_epoch,
                "i_batch": i_batch,
                "i_global": i_global,
            }
            lc_logger(dict_to_log, step)

            i_global += 1

        lc_logger.print(step)

    step = "Test"
    for i_batch, batch in enumerate(dl_tes):
        X, T = batch
        logits = model(X)
        pred_proba = softmax(logits)
        pred_class = pred_proba.argmax(dim=-1)
        loss = loss_func(logits, T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bACC = balanced_accuracy_score(T, pred_class)

        dict_to_log = {
            "loss_plot": float(loss),
            "balanced_ACC_plot": float(bACC),
            "pred_proba": pred_proba.detach().cpu().numpy(),
            "true_class": T.cpu().numpy(),
            "i_fold": i_fold,
            "i_epoch": i_epoch,
            "i_batch": i_batch,
            "i_global": i_global,
        }
        lc_logger(dict_to_log, step)

    lc_logger.print(step)

    # Plot using refactored method
    fig = lc_logger.plot_learning_curves(
        title=f"fold#{i_fold}",
        linewidth=1,
        scattersize=50,
        spath="learning_curve_logger_demo.jpg",
    )

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo learning curve logger")
    return parser.parse_args()


def run_main() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng

    import sys

    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
