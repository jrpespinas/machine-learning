"""Logistic Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "1.0.0"
__author__ = "Jan Rodolf Espinas"

import numpy as np

class LogisticRegression(self):
    def __init__(self):
        self.weights = None 
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

def main():
    pass

if __name__ == "__main__":
    main()