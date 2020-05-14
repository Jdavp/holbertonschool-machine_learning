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
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
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
        self.__weights = weights

    @property
    def weights(self):
        'A dictionary to hold all weights and biased of the network'
        return self.__weights

    @property
    def cache(self):
        'A dictionary to hold all intermediary values of the network'
        return self.__cache

    @property
    def L(self):
        'The number of layers in the neural network'
        return self.__L

    def sigmoid(self, z):
        "Apply sigmoid activation function"
        return 1/(1+np.exp(-z))

    def forward_prop(self, X):
        'Calculates the forward propagation of the neural network'
        self.__cache['A0'] = X

        for layer in range(self.__L):
            z = np.matmul(self.__weights['W'+str(layer+1)],
                          self.__cache['A'+str(layer)])\
                          + self.__weights['b'+str(layer+1)]
            actived = self.sigmoid(z)
            self.__cache['A'+str(layer+1)] = actived

        return actived, self.__cache
