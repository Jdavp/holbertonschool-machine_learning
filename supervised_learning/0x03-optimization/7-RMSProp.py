#!/usr/bin/env python3
'RMSProp'


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    'updates a variable using the RMSProp optimization algorithm'
    momentum = beta2 * s + (1 - beta2) * grad ** 2
    variable = var - alpha * grad / (momentum ** (1/2) + epsilon)
    return variable, momentum
