#!/usr/bin/env python3
'Pooling Back Prop'
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    'performs back propagation over a pooling layer of a neural network'
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    # In this notation row refers to height and col to width
    for img in range(m):
        A_img = A_prev[img]
        for height in range(h_new):
            for width in range(w_new):
                for ch in range(c_new):
                    # corners of the slice
                    row_start = height * sh
                    row_end = height * sh + kh
                    col_start = width * sw
                    col_end = width * sw + kw

                    slice_A = A_img[row_start:row_end, col_start:col_end, ch]
                    if mode == "max":
                        mask = (slice_A == np.max(slice_A)).astype(int)
                        aux = dA[img, height, width, ch] * mask
                        dA_prev[img, row_start:row_end, col_start:col_end,
                                ch] += aux
                    if mode == "avg":
                        average = dA[img, height, width, ch] / (kh * kw)
                        mask = np.ones(kernel_shape) * average
                        dA_prev[img, row_start:row_end, col_start:col_end,
                                ch] += mask
    return dA_prev
