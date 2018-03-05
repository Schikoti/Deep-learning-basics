import numpy as np
from scipy import stats
from sklearn import datasets

class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train=x
        self.y_train=y

    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """
        finallabel=[]
        for i in x:
            ED=[]
            for xtrain in self.x_train:
                Ed= np.linalg.norm(xtrain-i)
                ED.append(Ed)
            ids = np.argsort(ED)[:self.k]
            NL = np.take(self.y_train, ids)
            label= stats.mode(NL, axis=None)[0][0]
            finallabel.append(label)
        return np.asarray(finallabel)