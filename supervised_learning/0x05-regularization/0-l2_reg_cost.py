#!/usr/bin/env python3
'L2 Regularization Cost'
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    'calculates the cost of a neural network with L2 regularization'
    for layer in range(L):
        frob = np.linalg.norm(weights['W{}'.format(layer+1)])
        cost = cost + lambtha * frob / (2*m)
    return cost
