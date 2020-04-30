#!/usr/bin/env python3
"Initialize Exponential"


class Exponential:
    "represents a exponential distribution"

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
            self.lambtha = (1/data_mean)

    def factorial(self, n):
        'factorial of a number'
        if n == 0:
            return 1
        else:
            return n * self.factorial(n-1)

    def pdf(self, x):
        'Calculates the value of the PMF for a given number'

        if x < 0:
            return 0
        return self.lambtha * self.e ** -(self.lambtha * x)

    def cdf(self, k):
        'Calculates the value of the CDF for a given number'

        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k+1):
            cdf += (((self.lambtha ** i) / (self.factorial(i))) *
                    (self.e ** -self.lambtha))
        return cdf
