"""Linear Regression using Numpy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__: '1.0.0'
__author__: 'Jan Rodolf Espinas'

import numpy as np

# initialize sample data
data_size = 5 
theta = np.ones((2,1))
X = np.column_stack([np.ones(5), np.random.rand(5)])

def hypothesis(X,theta):
    return X @ theta

def cost_function():
    pass

def gradient_descent():
    pass

def main():
    print(hypothesis(X,theta))

if __name__ == '__main__':
    main()