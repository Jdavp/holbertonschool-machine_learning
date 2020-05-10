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

    def forward_prop(self, X):
        'Calculates the forward propagation of the neuron'
        hidden_layer = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(hidden_layer)
        return self.__A

    def sigmoid(self, z):
        "Apply sigmoid activation function"
        return 1/(1+np.exp(-z))

    def cost(self, Y, A):
        "cost of the model using logistic regression"
        lost = (np.multiply(np.log(A), Y) +
                np.multiply(np.log(1.0000001-A), (1-Y)))
        m = Y.shape[1]
        cost = -np.sum(lost)/m
        return cost

    def evaluate(self, X, Y):
        'Evaluates the neuron’s predictions'
        A = np.array(self.forward_prop(X))
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return pred, cost
