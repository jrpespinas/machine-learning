# Copyright 2020 Jan Rodolf Espinas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Linear Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '2.0.0'
__author__ = 'Jan Rodolf Espinas'

import numpy as np
from sklearn.datasets import load_boston


np.random.seed(42)


class LinearRegression():

    def __init__(self):
        self.theta = None

    def fit(self, features, y, alpha=0.001, epochs=1000, verbosity=False):
        '''
        This function uses gradient descent to minimize the error of the loss
        function therefore finding the best line that fits through the dataset.

        Parameters
        ----------
        features : numpy.ndarray
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

        # Append vector of `ones` before the first column of features
        m, n = features.shape
        X = np.ones((m, n+1))
        X[:, 1:] = features
        print(X)

        # initialize the weights
        theta = np.random.randn(n+1)
        self.theta = theta[:, np.newaxis]

        # GRADIENT DESCENT
        for i in range(epochs):
            y_pred = self.predict(X)
            self.theta = self.theta - alpha * (1/m) * X.T.dot((y_pred - y))

            # Display weights and loss per epoch
            if verbosity:
                if i % 100 == 0:
                    loss = self.loss(X, y)
                    print(f'epoch {i}: loss: {loss}, weights: {self.theta}, ')

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
        return np.dot(X, self.theta)

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
    y = dataset.target[:, np.newaxis]
    RM = features[:, 5]
    RM = RM[:, np.newaxis]
    DIS = features[:, 7]
    DIS = DIS[:, np.newaxis]
    X = np.ones((DIS.shape[0], 2))
    X[:, 0:] = RM
    X[:, 1:] = DIS

    model = LinearRegression()
    model.fit(RM, y, alpha=0.02, verbosity=1)


if __name__ == '__main__':
    main()
