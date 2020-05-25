#!/usr/bin/env python3
'Batch Normalization'
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    'normalizes unactivated output of using batch normalization'
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    z_norm = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * z_norm + beta
