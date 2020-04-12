"""Linear Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__: '1.0.0'
__author__: 'Jan Rodolf Espinas'

import numpy as np
from sklearn.datasets import load_boston

# DATASET
dataset = load_boston()
features = dataset.data
y = dataset.target[:,np.newaxis] 

def hypothesis(X,theta):
    return X @ theta

def cost_function(X,y,theta):
    pass

def gradient_descent():
    pass

def main():
    print(features)
    print(y)

if __name__ == '__main__':
    main()