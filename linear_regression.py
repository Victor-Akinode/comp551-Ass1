import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y, lam = 0):

        y = y.reshape(-1)
        X = np.c_[np.ones(len(X)), X]

        if lam > 0:

            I = np.eye(X.shape[1])
            I[0, 0] = 0

            X = np.vstack([X, np.sqrt(lam) * I])
            y = np.concatenate([y, np.zeros(X.shape[1])])

        self.w = np.linalg.lstsq(X, y, rcond = None)[0]

    def predict(self, X):

        if self.w is None:
            raise ValueError("Model not fitted.")
        
        X = np.c_[np.ones(len(X)), X]
        return X @ self.w