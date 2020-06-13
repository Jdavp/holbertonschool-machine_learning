#!/usr/bin/env python3
'Pooling Forward Prop'
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    'performs forward propagation over a pooling layer of a neural network'

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pool_h = (h_prev - kh) // sh + 1
    pool_w = (w_prev - kw) // sw + 1

    pooled = np.zeros((m, pool_h, pool_w, c_prev))
    for height in range(pool_h):
        for width in range(pool_w):
            slice_A = A_prev[:, height * sh:height * sh + kh,
                             width * sw:width * sw + kh]
            if mode == "max":
                pooled[:, height, width] = np.max(slice_A, axis=(1, 2))
            if mode == "avg":
                pooled[:, height, width] = np.mean(slice_A, axis=(1, 2))
    return pooled
