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

    def evaluate(self, X, Y):
        'Evaluates the neuron’s predictions'
        A = np.array(self.forward_prop(X))
        pred = np.where(A[1] >= 0.5, 1, 0)
        cost = self.cost(Y, A[1])
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        'Calculates one pass of gradient descent on the neuron'
        m2 = A2.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / m2
        db2 = np.sum(dz2, axis=1, keepdims=True) / m2
        m1 = A1.shape[1]
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(X, dz1.T) / m1
        db1 = np.sum(dz1, axis=1, keepdims=True) / m1
        self.__W1 = self.__W1 - (alpha * dw1.T)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2.T)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        'trains neuron'
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for it in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2,  alpha)
        return self.evaluate(X, Y)
