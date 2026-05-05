#!/usr/bin/env python3

import numpy as np


def _connect_nums(values):
    """Local stand-in for ``scitex.gen.connect_nums`` — joins a tuple of
    ints/strings into a hyphen-separated label, e.g. (1, 2) -> "1-2".

    Inlined here so this module has zero dependency on the umbrella
    ``scitex`` package. If we later need the full ``scitex.gen.connect_nums``
    behaviour (range collapsing), reach into ``scitex_gen`` once that
    package exists.
    """
    return "-".join(str(v) for v in values)


def merge_labels(*ys, to_int=False):
    if not len(ys) > 1:  # Check if more than two arguments are passed
        return ys[0]
    else:
        y = [_connect_nums(zs) for zs in zip(*ys)]
        if to_int:
            conv_d = {z: i for i, z in enumerate(np.unique(y))}
            y = [conv_d[z] for z in y]
        return np.array(y)
