"""Linear Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__: '1.0.0'
__author__: 'Jan Rodolf Espinas'

import numpy as np
from sklearn.datasets import load_boston

def hypothesis(X,theta):
    return X @ theta

def cost_function(X,y,theta):
    pass

def gradient_descent():
    pass

def main():
    # DATASET
    dataset = load_boston()
    features = dataset.data
    y = dataset.target[:,np.newaxis] 

    RM = features[:,5] 

    # Create matrix of features
    X = np.ones((RM.shape[0],2))
    X[:,1:] = RM[:,np.newaxis]

    # Create matrix of theta
    theta = np.random.randn(2)
    theta = theta[:,np.newaxis]

    print(X)
    print(theta)

if __name__ == '__main__':
    main()