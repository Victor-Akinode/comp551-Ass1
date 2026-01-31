import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, X):
        return X @ self.w
