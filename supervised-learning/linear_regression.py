"""Linear Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '3.0.0'
__author__ = 'Jan Rodolf Espinas'

import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 1

    def initialize_weights(self, X):
        return np.random.randn(X.shape[1], 1)

    def predict(self, X):
        return np.dot(self.weights, X) + self.bias

    def loss(self, y, y_hat):
        return np.sum(np.power((y_hat - y), 2)) / (2 * len(y))

    def fit(self, X, y, learning_rate, epochs, verbosity: bool = True):
        self.weights = self.initialize_weights(X)

        for epoch in epochs:
            y_hat = self.predict(X)

            pd_w = (1 / X.shape[0]) + np.dot(X, (y_hat - y))
            pd_b = (1 / X.shape[0]) + np.sum(y_hat - y)

            loss = self.loss(y, y_hat)

            self.weights -= learning_rate * pd_w
            self.bias -= learning_rate * pd_b

            if verbosity:
                print("Epoch: {}, Loss: {}".format(epoch, loss))
