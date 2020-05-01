#!/usr/bin/env python3
"Initialize Normal"


class Normal:
    "represents a exponential distribution"

    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        'Normal contructor'
        self.data = data

        if stddev <= 0:
            raise ValueError('stddev must be a positive value')
        self.stddev = float(stddev)
        self.mean = float(mean)
        if data is None:
            self.mean = self.mean
            self.stddev = self.stddev
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            sigma = 0
            i = 1
            for i in range(len(data)):
                sigma += (data[i] - self.mean) ** 2
            variance = (1/len(data)) * sigma
            self.stddev = (variance ** (1/2))

    def z_score(self, x):
        'Calculates the z-score of a given x-value'
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        'Calculates the x-value of a given z-score'
        return self.mean + (z * self.stddev)

    def factorial(self, n):
        'factorial of a number'
        if n == 0:
            return 1
        else:
            return n * self.factorial(n-1)

    def pdf(self, x):
        'Calculates the value of the PDF of a given x-value'

        raiz = (((2 * ((self.π)*(self.stddev**2)))) ** (1/2))
        potencia_1 = -((x - self.mean) ** 2)
        potencia_2 = (2 * (self.stddev ** 2))
        return ((1/raiz) * (self.e ** ((potencia_1) / potencia_2)))

    def cdf(self, x):
        'Calculates the value of the CDF for a given time period'

        if x < 0:
            return 0
        return 1 - (self.e ** - (self.lambtha * x))
