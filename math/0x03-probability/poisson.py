#!/usr/bin/env python3
"Initialize Poisson"


class Poisson:
    "represents a poisson distribution"

    def __init__(self, data=None, lambtha=1.):
        'Poisson contructor'

        if lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        self.lambtha = float(lambtha)
        if data is None:
            data = self.lambtha
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            data_mean = sum(data) / len(data)
            self.lambtha = data_mean
