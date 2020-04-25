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
        self.theta = None

    def loss(self, X, y):
        pass

    def fit(self, X, y, alpha=0.001, epochs=1000, verbosity=False):
        pass

    def predict(self, X):
        '''
        Also known as the hypothesis, this function takes in 
        the input matrix `X` and produces an output.

        Parameters
        ----------
        X : numpy.ndarray
            Matrix containing the inputs
        
        Returns
        -------
        numpy.ndarray
            Matrix containing the outputs
        '''
        return X @ self.theta

def main():
    dataset = load_boston()
    features = dataset.data
    y = dataset.target[:,np.newaxis] 
    RM = features[:,5]
    RM = RM[:,np.newaxis]
    print(RM.shape)

    model = LinearRegression()
    #model.fit(RM, y, alpha=0.02, verbosity=1)

if __name__ == '__main__':
    main()