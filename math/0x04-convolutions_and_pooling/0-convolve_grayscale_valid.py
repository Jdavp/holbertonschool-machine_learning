#!/usr/bin/env python3
'Valid Convolution'
import numpy as np

    
def convolve_grayscale_valid(images, kernel):
    ' performs a valid convolution on grayscale images'
    m = images.shape[0]
    kh = kernel.shape[0]    
    kw = kernel.shape[1]
    cm_h = images.shape[1] - kh + 1
    cm_w = images.shape[2] - kw + 1
    conv_m = np.zeros((m, cm_h, cm_w))
    for row_idx in range(cm_h):
        for col_idx in range(cm_w):
            mul_idx = images[:, row_idx:row_idx + kh, col_idx:col_idx + kw] * kernel
            sum_idx = np.sum(mul_idx, axis=(1,2))
            conv_m[:, row_idx, col_idx] = sum_idx
    return conv_m