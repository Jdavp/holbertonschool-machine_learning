#!/usr/bin/env python3
'DeepNeuralNetwork'
import numpy as np


class DeepNeuralNetwork:
    'deep neural network performing binary classification'

    def __init__(self, nx, layers):
        'class constructor'

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        weights = {}
        for la in range(len(layers)):
            if layers[la] < 1:
                raise TypeError("layers must be a list of positive integers")
            w_key = 'W'+str(la + 1)
            b_key = 'b'+str(la + 1)
            if la == 0:
                weights[w_key] = np.random.randn(layers[la], nx)*np.sqrt(2/nx)
            else:
                weights[w_key] = np.random.randn(layers[la], layers[la-1]) *\
                                 np.sqrt(2 / layers[la-1])
            weights[b_key] = np.zeros((layers[la], 1))
        self.weights = weights
