#!/usr/bin/env python3
'Gradient Descent with L2 Regularization'
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    'using gradient descent with L2 regularization'
    m = Y.shape[1]

    for layer in reversed(range(L)):
        A = cache['A' + str(layer + 1)]
        A_dw = cache['A' + str(layer)]
        if layer == L - 1:
            dz = A - Y
            W = weights['W' + str(layer + 1)]
        else:
            da = 1 - (A * A)
            dz = np.matmul(W.T, dz)
            dz = dz * da
            W = weights['W' + str(layer + 1)]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(layer + 1)] = (weights['W'+str(layer+1)] -
                                         alpha * (dw.T +
                                         (lambtha / m *
                                          weights['W'+str(layer+1)])))
        weights['b' + str(layer + 1)] = (weights['b'+str(layer+1)] -
                                         alpha * db)
