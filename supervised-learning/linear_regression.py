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
    return sum(np.square((y - hypothesis(X,theta))))/X.shape[0]

def gradient_descent(X,y,theta,alpha,iterations):
    for i in range(iterations):
        predictions = hypothesis(X,theta)
        theta = theta - (alpha/len(X))*(-X.T.dot(y-predictions))

    if i % 50 == 0:
        print(f'Error in {i}th iteration: {cost_function(X,y,theta)}')

def main():
    # DATASET
    dataset = load_boston()
    features = dataset.data
    y = dataset.target[:,np.newaxis] 

    # This feature was particularly chosen as the variable for univariate
    # linear regression after visualizing the dataset in a separate notebook.
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