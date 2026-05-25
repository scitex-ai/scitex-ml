#!/usr/bin/env python3
"""Optimizer utilities using external packages."""

import torch.optim as optim

from scitex_dev import try_import_optional

# Use pytorch-optimizer package for Ranger when available; fall back
# to the vendored Ranger_Deep_Learning_Optimizer copy if not installed.
# Neither is declared as an extra — pytorch_optimizer is an optional
# convenience dep; omit `extra=`/`pkg=` per the playbook rule.
Ranger = try_import_optional("pytorch_optimizer", "Ranger21")
if Ranger is None:
    Ranger = try_import_optional(
        ".Ranger_Deep_Learning_Optimizer.ranger.ranger2020",
        "Ranger",
        package=__package__,
    )
RANGER_AVAILABLE = Ranger is not None


def get_optimizer(name: str):
    """Get optimizer class by name.

    Args:
        name: Optimizer name (adam, ranger, rmsprop, sgd)

    Returns:
        Optimizer class

    Raises:
        ValueError: If optimizer name is not supported
    """
    optimizers = {"adam": optim.Adam, "rmsprop": optim.RMSprop, "sgd": optim.SGD}

    if name == "ranger":
        if not RANGER_AVAILABLE:
            raise ImportError(
                "Ranger optimizer not available. "
                "Please install pytorch-optimizer: pip install pytorch-optimizer"
            )
        optimizers["ranger"] = Ranger

    if name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}"
        )

    return optimizers[name]


def set_optimizer(models, optimizer_name: str, lr: float):
    """Set optimizer for models.

    Args:
        models: Model or list of models
        optimizer_name: Name of optimizer
        lr: Learning rate

    Returns:
        Configured optimizer instance
    """
    if not isinstance(models, list):
        models = [models]

    learnable_params = []
    for model in models:
        learnable_params.extend(list(model.parameters()))

    optimizer_class = get_optimizer(optimizer_name)
    return optimizer_class(learnable_params, lr=lr)
