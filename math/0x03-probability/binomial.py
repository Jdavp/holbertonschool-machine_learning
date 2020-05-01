#!/usr/bin/env python3
"Initialize Binomial"


class Binomial:
    "represents a binomial distribution"

    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, n=1, p=0.5):
        'Binomial contructor'
        if n <= 0:
            raise ValueError('n must be a positive value')
        if p < 0 or p > 1:
            raise ValueError('p must be greater than 0 and less than 1')
        self.n = int(n)
        self.p = float(p)
        if data is None:
            self.n = self.n
            self.p = self.p
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

    def pmf(self, k):
        'Calculates the value of the PMF for a given number'

        raiz = (((2 * ((self.π)*(self.stddev**2)))) ** (1/2))
        potencia_1 = -((x - self.mean) ** 2)
        potencia_2 = (2 * (self.stddev ** 2))
        return ((1/raiz) * (self.e ** ((potencia_1) / potencia_2)))

    def cdf(self, k):
        'Calculates the value of the PMF for a given number'

        x_func = ((x - self.mean)/(self.stddev * (2 ** (1/2))))
        error_division = ((2 / (self.π * (1/2))))
        power_three = ((x_func ** 3)/3)
        power_five = ((x_func ** 5)/10)
        power_seven = ((x_func ** 7)/42)
        power_nine = ((x_func ** 9)/216)
        error_function = (error_division *
                          (x_func - power_three +
                           power_five - power_seven + power_nine))
        return ((1/2) * (1 + error_function))
