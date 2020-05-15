#!/usr/bin/env python3
'DeepNeuralNetwork'
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def cost(self, Y, A):
        "cost of the model using logistic regression"
        lost = (np.multiply(np.log(A), Y) +
                np.multiply(np.log(1.0000001-A), (1-Y)))
        m = Y.shape[1]
        cost = -np.sum(lost)/m
        return cost

    def evaluate(self, X, Y):
        'Evaluates the neural network’s predictions'
        A, _ = np.array(self.forward_prop(X))
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        'Calculates one pass of gradient descent on the neural network'
        m = Y.shape[1]
        for layer in reversed(range(self.__L)):
            A = cache['A' + str(layer + 1)]
            A_dw = cache['A' + str(layer)]
            if layer == self.__L - 1:
                dz = A - Y
                W = self.__weights['W' + str(layer + 1)]
            else:
                da = A * (1 - A)
                dz = np.matmul(W.T, dz)
                dz = dz * da
            W = self.__weights['W' + str(layer + 1)]
            dw = np.matmul(A_dw, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights['W' + str(layer + 1)] = (self.__weights
                                                    ['W'+str(layer+1)]
                                                    - alpha * dw.T)
            self.__weights['b' + str(layer + 1)] = (self.__weights
                                                    ['b'+str(layer+1)]
                                                    - alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        'Trains the neural network'
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if type(step)is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        cost_list = []
        for index, iteration in enumerate(range(iterations+1)):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost_list.append(self.cost(Y, A))
            if verbose is True and iteration % step == 0:
                print("Cost after {} iterations: {}"
                      .format(iteration, cost_list[iteration]))
        'graph'
        if graph is True:
            plt.title("Training Cost")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.plot(np.arange(0, iterations + 1), cost_list)
        return self.evaluate(X, Y)
        
    def save(self, filename):
        'Saves the instance object to a file in pickle format'
        if filename[-3:] is not ".pkl":
            filename = str(filename)+".pkl"
        return pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        'Loads a pickled DeepNeuralNetwork object'
        try:
            return pickle.load(open(filename, "rb"))
        except FileNotFoundError:
            return None
