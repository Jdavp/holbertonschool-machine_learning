#!/usr/bin/env python3
'DeepNeuralNetwork'
import numpy as np


class DeepNeuralNetwork:
    'deep neural network performing binary classification'

    def __init__(self, nx, layers):
        'class constructor'

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx <= 0:
            raise ValueError('nx must be a positive integer')

        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')

        if min(layers) <= 0:
            raise TypeError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        weight = {}
        count_l = nx
        for l in range(len(layers)):
            if type(layers[l]) is not int:
                raise ValueError('layers must be a list of positive integers')
            w_key = "W"+str(l + 1)
            weight[w_key] = np.random.randn(layers[l], count_l)*np.sqrt(2 / nx)
            b_key = "b"+str(l + 1)
            weight[b_key] = np.zeros((layers[l], 1))
            count_l = layers[l]
        self.weights = weight
