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
            raise ValueError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        weight = {}
        count_l = nx
        for layer in range(len(layers)):
            weight["W"+str(layer + 1)] = np.random.randn(layers[layer],
                                                         count_l)\
                                                         * np.sqrt(2/nx)
            weight["b"+str(layer + 1)] = np.zeros((layers[layer], 1))
            count_l = layers[layer]
        self.weights = weight
