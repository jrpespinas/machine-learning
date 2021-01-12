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
        self.bias = 0

    def initialize_weights(self, X):
        num_features = X.shape[1]
        return np.zeros(num_features)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def loss(self, y, y_hat):
        return np.sum(np.power((y_hat - y), 2)) / (2 * len(y))

    def fit(self, X, y, learning_rate: float = 0.1,
            epochs: int = 300, verbosity: bool = True):
        self.weights = self.initialize_weights(X)
        num_samples = X.shape[0]

        for epoch in range(epochs):
            y_hat = self.predict(X)

            pd_w = (1 / num_samples) * np.dot(X.T, (y_hat - y))
            pd_b = (1 / num_samples) * np.sum(y_hat - y)

            loss = self.loss(y, y_hat)

            self.weights -= learning_rate * pd_w
            self.bias -= learning_rate * pd_b

            if verbosity:
                print("Epoch: {}, Loss: {}".format(epoch, loss))
