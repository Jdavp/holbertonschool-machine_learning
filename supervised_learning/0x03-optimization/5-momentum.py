#!/usr/bin/env python3
'Momentum'


def update_variables_momentum(alpha, beta1, var, grad, v):
    'updates a variable using gradient descent momentum optimization algorithm'

    momentum = beta1 * v + (1 - beta1) * grad
    variable = var - alpha * momentum

    return variable, momentum
