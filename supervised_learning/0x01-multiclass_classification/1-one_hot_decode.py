#!/usr/bin/env python3
'One-Hot Decode '

import numpy as np


def one_hot_decode(one_hot):
    'converts a one-hot matrix into a vector of labels'
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot) == 0 or len(one_hot.shape) != 2:
        return None
    new = np.argmax(one_hot, axis=0)
    return new
