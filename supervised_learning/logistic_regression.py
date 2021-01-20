"""Logistic Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Jan Rodolf Espinas"

import numpy as np

class LogisticRegression(self):
    def __init__(self):
        self.weights = None 
        self.bias = 0
    
    def initialize_weights(self, X):
        num_features = X.shape[1]
        return np.zeros(num_features)

    def predict(self, X):
        return np.dot(self.weights, X) + self.bias
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_hat):
        m = y.shape[1]
        
        loss = np.multiply(y, np.log(y_hat)) + \
            np.multiply((1 - y), np.log(1 - y_hat))
        total_loss = - np.sum(loss) / m

        total_loss = np.squeeze(total_loss)

        return total_loss

def main():
    pass

if __name__ == "__main__":
    main()