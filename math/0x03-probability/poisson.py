#!/usr/bin/env python3
"Initialize Poisson"


class Poisson:
    "represents a poisson distribution"

    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        'Poisson contructor'
        self.data = data

        if lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        self.lambtha = float(lambtha)
        if data is None:
            self.data = self.lambtha
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            data_mean = sum(data) / len(data)
            self.lambtha = data_mean

    def factorial(self, n):
        'factorial of a number'
        if n == 0:
            return 1
        else:
            return n * self.factorial(n-1)

    def pmf(self, k):
        'Calculates the value of the PMF for a given number'

        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        return (((self.lambtha ** k) / (self.factorial(k))) *
                (self.e ** -self.lambtha))
