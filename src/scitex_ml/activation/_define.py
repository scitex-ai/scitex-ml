try:
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "scitex_ml.activation requires torch: pip install scitex-ml[heavy]"
    ) from exc


def define(act_str):
    acts_dict = {
        "relu": nn.ReLU(),
        "swish": nn.SiLU(),
        "mish": nn.Mish(),
        "lrelu": nn.LeakyReLU(0.1),
    }
    return acts_dict[act_str]
