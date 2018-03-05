import numpy as np


class LeastSquares(object):
    def __init__(self, k):
        """
        Initialize the LeastSquares class

        The input parameter k specifies the degree of the polynomial
        """
        self.k = k
        self.coeff = None

    def fit(self, x, y):
        """
        Find coefficients of polynomial that predicts y given x with
        degree self.k

        Store the coefficients in self.coeff
        """
        k = self.k
        A = np.vander(x, N=k+1, increasing = True)
        AT = np.linalg.pinv(A)
        self.coeff = np.dot(AT, y)


    def predict(self, x):
        """
        Predict the output given x using the learned coeffecients in
        self.coeff
        """
        k = self.k
        A = np.vander(x, N=k + 1, increasing = True)
        y = np.dot(A, self.coeff)
        return y
