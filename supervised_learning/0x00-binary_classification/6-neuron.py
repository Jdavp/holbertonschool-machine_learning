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
        z = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        'Calculates one pass of gradient descent on the neuron'
        m = A.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        'trains neuron'
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be an float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for it in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
