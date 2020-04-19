"""Linear Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__: '2.0.0'
__author__: 'Jan Rodolf Espinas'

import numpy as np
from sklearn.datasets import load_boston

class LinearRegression():

    def __init__(self):
        self.weights = None
        self.bias = 0

    def loss(self, X, y, theta):
        pass

    def fit(self, X, y, alpha=0.001, epochs=1000, verbosity=False):
        pass

    def predict(self, X):
        pass

def main():
    pass

if __name__ == '__main__':
    main()