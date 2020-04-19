"""Linear Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__: '2.0.0'
__author__: 'Jan Rodolf Espinas'

import numpy as np
from sklearn.datasets import load_boston

np.random.seed(42)

class LinearRegression():

    def __init__(self):
        self.weights = None
        self.bias = 1

    def _loss(self):
        pass

    def fit(self, X, y, alpha=0.001, epochs=1000, verbosity=False):
        X = X[:,np.newaxis]
        m, features = X.shape
        
        # initialize weights
        self.weights = np.random.randn(features,1)

        # hypothesis function
        hypothesis = self.bias + (X @ self.weights)
        

    def predict(self, X):
        pass

def main():
    dataset = load_boston()
    features = dataset.data
    y = dataset.target[:,np.newaxis] 
    RM = features[:,5] 

    model = LinearRegression()
    model.fit(RM,y)

if __name__ == '__main__':
    main()