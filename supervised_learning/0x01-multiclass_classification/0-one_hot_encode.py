#!/usr/bin/env python3
'One-Hot Encode'
import numpy as np


def one_hot_encode(Y, classes):
    'converts a numeric label vector into a one-hot matrix'

    if type(classes) is not int or type(Y) is not np.ndarray:
        return None
    if classes is None or Y is None:
        return None
    try:
        encode = np.zeros((classes, len(Y)))
        fila = np.arange(Y.shape[0])
        encode[Y, fila] = 1
        return encode
    except Exception:
        return None
