#!/usr/bin/env python3
'NeuralNetwork'
import numpy as np


class NeuralNetwork:
    'neural network with one hidden layer performing binary classification'

    def __init__(self, nx, nodes):
        'class constructor'

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx <= 0:
            raise ValueError('nx must be a positive integer')

        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes <= 0:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        "output"
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        'get method return private instance atribute W'
        return self.__W1

    @property
    def b1(self):
        'get method return private instance atribute b'
        return self.__b1

    @property
    def A1(self):
        'get method return private instance atribute A'
        return self.__A1

    @property
    def W2(self):
        'get method return private instance atribute W'
        return self.__W2

    @property
    def b2(self):
        'get method return private instance atribute b'
        return self.__b2

    @property
    def A2(self):
        'get method return private instance atribute A'
        return self.__A2

    def sigmoid(self, z):
        "Apply sigmoid activation function"
        return 1/(1+np.exp(-z))

    def forward_prop(self, X):
        'Calculates the forward propagation of the neural network'
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        "cost of the model using logistic regression"
        lost = (np.multiply(np.log(A), Y) +
                np.multiply(np.log(1.0000001-A), (1-Y)))
        m = Y.shape[1]
        cost = -np.sum(lost)/m
        return cost
