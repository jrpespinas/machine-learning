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

    def fit(self, X, y, alpha=0.001, epochs=1000, verbosity=False):
        '''
        This function uses gradient descent to minimize the error of the loss
        function therefore finding the best line that fits through the dataset.

        Parameters
        ----------
        X : numpy.ndarray
            2-dimensional matrix containing the inputs
        y : numpy.ndarray
            Nx1 matrix containing the actual values
        alpha : float
            Learning rate
        epochs : int 
            Number of iterations
        verbosity : bool 
            Display the weights and loss per epoch
        '''
        

    def predict(self, X):
        '''
        Also known as the hypothesis, this function takes in 
        the input matrix `X` and produces an output.

        Parameters
        ----------
        X : numpy.ndarray
            2-dimensional matrix containing the inputs
        
        Returns
        -------
        numpy.ndarray
            Matrix containing the outputs
        '''
        return X @ self.theta

    def loss(self, X, y):
        '''
        This function calculates the performance of the model

        Parameters
        ----------
        X : numpy.ndarray
            2-dimensional matrix containing the inputs
        y : numpy.ndarray
            Nx1 matrix containing the actual values

        Returns
        -------
        float 
            The loss value
        '''
        m, _ = X.shape
        return (1 / (2*m)) * sum(np.square(self.predict(X)-y))

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