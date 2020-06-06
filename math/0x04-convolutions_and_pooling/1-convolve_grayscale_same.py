#!/usr/bin/env python3
'Same Convolution'
import numpy as np


def convolve_grayscale_same(images, kernel):
    'performs a same convolution on grayscale images'
    m = images.shape[0]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    cm_h = images.shape[1]
    cm_w = images.shape[2]

    if kh % 2 != 0:
        pad_h = (kh - 1) // 2
    else:
        pad_h = kh // 2
    if kw % 2 != 0:
        pad_w = (kw - 1) // 2
    else:
        pad_w = kw // 2

    padding = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    conv_m = np.zeros((m, cm_h, cm_w))
    for row_idx in range(cm_h):
        for col_idx in range(cm_w):
            mul_idx = padding[:, row_idx:row_idx + kh,
                              col_idx:col_idx + kw] * kernel
            sum_idx = np.sum(mul_idx, axis=(1, 2))
            conv_m[:, row_idx, col_idx] = sum_idx
    return conv_m
