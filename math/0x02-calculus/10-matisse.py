#!/usr/bin/env python3
'Derive happiness in oneself from a good days work'


def poly_derivative(poly):
    'calculates the derivative of a polynomial'
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]

    return [poly[i] * i for i in range(1, len(poly))]
