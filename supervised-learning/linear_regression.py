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
        self.bias = None

    def predict(self, X):
        return np.dot(self.weights, X) + self.bias

    def loss(self, y, y_hat):
        return np.sum(np.power((y_hat - y), 2)) / (2 * len(y))
