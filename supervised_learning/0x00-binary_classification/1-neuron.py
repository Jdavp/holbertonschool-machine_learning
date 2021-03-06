#!/usr/bin/env python3
'Neuron'
import numpy as np


class Neuron:
    'defines a single neuron performing binary classification'

    def __init__(self, nx):
        'class constructor'

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx <= 0:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.normal(size=[1, nx])
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        'get method return private instance atribute W'
        return self.__W

    @property
    def b(self):
        'get method return private instance atribute b'
        return self.__b

    @property
    def A(self):
        'get method return private instance atribute A'
        return self.__A
