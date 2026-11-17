from scitex_ml._optional import missing_optional_dependency

try:
    import torch.nn as nn
except ImportError as _exc:
    raise missing_optional_dependency("scitex_ml.activation", _exc) from _exc


def define(act_str):
    acts_dict = {
        "relu": nn.ReLU(),
        "swish": nn.SiLU(),
        "mish": nn.Mish(),
        "lrelu": nn.LeakyReLU(0.1),
    }
    return acts_dict[act_str]
