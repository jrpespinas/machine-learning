import numpy as np
from linear_regression import LinearRegression
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
X = X[:, 4]
X = X[:, np.newaxis]

model = LinearRegression()
model.fit(X, y, 0.1, 300)
