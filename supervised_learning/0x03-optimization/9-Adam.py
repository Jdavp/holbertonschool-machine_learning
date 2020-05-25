#!/usr/bin/env python3
'Adam'
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    'updates a variable in place using the Adam optimization algorithm'

    vdw = beta1 * v + (1 - beta1) * grad
    vds = beta2 * s + (1 - beta2) * grad ** 2

    correct_vdw = vdw / (1 - beta1 ** t)
    correct_vds = vds / (1 - beta2 ** t)
    var = var - alpha * (correct_vdw / (np.sqrt(correct_vds) + epsilon))

    return var, vdw, vds
