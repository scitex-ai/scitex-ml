#!/usr/bin/env python3
"""Scitex utils module."""

from scitex_dev import try_import_optional

from ._check_params import check_params
from ._format_samples_for_sktime import format_samples_for_sktime
from ._merge_labels import merge_labels
from ._sliding_window_data_augmentation import sliding_window_data_augmentation
from ._under_sample import under_sample

# torch-dependent modules ([heavy] extra) — gate them so that
# `import scitex_ml` succeeds without torch installed.
DefaultDataset = try_import_optional(
    "._default_dataset",
    attr="DefaultDataset",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)

LabelEncoder = try_import_optional(
    "._label_encoder",
    attr="LabelEncoder",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)

verify_n_gpus = try_import_optional(
    "._verify_n_gpus",
    attr="verify_n_gpus",
    extra="heavy",
    pkg="scitex-ml",
    package=__name__,
)

__all__ = [
    "DefaultDataset",
    "LabelEncoder",
    "check_params",
    "format_samples_for_sktime",
    "merge_labels",
    "sliding_window_data_augmentation",
    "under_sample",
    "verify_n_gpus",
]
