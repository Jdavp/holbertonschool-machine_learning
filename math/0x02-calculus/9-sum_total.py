#!/usr/bin/env python3
'Our life is the sum total of all the decisions we make every day,'\
    'and those decisions are determined by our priorities'


def summation_i_squared(n):
    "that calculates sum_{i=1}^{n} i^2"

    if not isinstance(n, int):
        return None

    if n < 1:
        return None
    return int((n / 6) * (n + 1) * (2 * n + 1))
