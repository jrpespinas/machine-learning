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

    def _loss(self, m, hypothesis, y):
        return (1 / (2*m)) * sum(np.square(hypothesis - y))

    def fit(self, X, y, alpha=0.001, epochs=1000, verbosity=False):
        X = X[:,np.newaxis]
        m, features = X.shape
        
        # Initialize weights
        self.weights = np.random.randn(features,1)

        # Gradient Descent
        for i in range(epochs):

            # Hypothesis Function
            hypothesis = self.bias + (X @ self.weights)

            # Partial Derivatives of the Cost Function
            pd_b = (1/m) * sum(hypothesis-y)
            pd_w = (1/m) * np.dot(X.T, (hypothesis-y))

            # Assigning theta
            self.bias -=  alpha*pd_b
            self.weights -= alpha*pd_w

            # Get loss
            loss = self._loss(m, hypothesis, y)

            # display loss and weights
            if verbosity:
                print(f'Epoch: {i}, Weights: {self.bias[0]}, Loss: {loss}')

    def predict(self, X):
        return self.bias + (X @ self.weights)

def main():
    dataset = load_boston()
    features = dataset.data
    y = dataset.target[:,np.newaxis] 
    RM = features[:,5] 

    model = LinearRegression()
    model.fit(RM, y, epochs=1, verbosity=1)

if __name__ == '__main__':
    main()